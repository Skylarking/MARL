import numpy as np
import torch.nn as nn
import torch.nn.functional as f
import torch as th
from network.mixer import VDNMixer, QMixMixer, DMAQer
import copy
import os


class QLearner():
    def __init__(self, mac, args):
        # other args
        self.max_episode_len = args.episode_limit
        self.gamma = args.gamma
        self.lr = args.lr
        self.model_dir = args.model_dir + '/' + args.alg + '/' + args.map
        self.args = args

        # network and params
        self.eval_net = mac
        self.target_net = copy.deepcopy(mac)    # target network to calculate the ground truth
        self.params = list(mac.parameters())

        # TODO begin 这里修改Mixer网络，可以改成QMIX、QPLEX等算法
        if args.alg == 'vdn':
            self.mixer = VDNMixer(args)
        elif args.alg == 'qmix':
            self.mixer = QMixMixer(args)
        elif args.alg == 'qplex':
            self.mixer = DMAQer(args)
        else:
            raise ValueError("Mixer {} not recognised.".format(args.alg))
        # TODO end
        self.target_mixer = copy.deepcopy(self.mixer)
        self.params += list(self.mixer.parameters())

        # model cuda
        if args.cuda:
            self.cuda()

        # optimizer
        if args.optimizer == "RMS":
            self.optimizer = th.optim.RMSprop(self.params, lr=self.lr)
        elif args.optimizer == "Adam":
            self.optimizer = th.optim.Adam(self.params, lr=self.lr)
        else:
            raise ValueError("optimizer {} not recognised.".format(args.optimizer))

    def get_max_episode_len(self, batch):
        # calculate the max_episode_len of n episodes for cutting redundant transitions
        n_episodes = batch["o"].shape[0]
        max_episode_len = 0
        for episode_idx in range(n_episodes):
            for transition_idx in range(self.args.episode_limit):
                if batch['terminated'][episode_idx][transition_idx][0] == 1:
                    if transition_idx + 1 >= max_episode_len:
                        max_episode_len = transition_idx + 1
                    break
        if max_episode_len == 0:  # 防止所有episode都是最大长度
            max_episode_len = self.args.episode_limit

        # cut redundant transitions
        for key in batch.keys():
            batch[key] = batch[key][:, :max_episode_len]  # 截取max_episode_len前面的数据

        return batch, max_episode_len

    def train(self, batch, train_step):
        # get the new batch & max_len
        episode_num = batch["o"].shape[0]
        batch, self.max_episode_len = self.get_max_episode_len(batch)

        # batch to tensor
        for key in batch.keys():  # 把batch里的数据转化成tensor
            if key == 'u':
                batch[key] = th.tensor(batch[key], dtype=th.long)   # action dtype is long
            else:
                batch[key] = th.tensor(batch[key], dtype=th.float32)

        # get the batch needed
        s, u, r, s_next, avail_u, avail_u_next, terminated = batch['s'], batch['u'], batch['r'], batch['s_next'], batch['avail_u'], batch['avail_u_next'], batch['terminated']
        u_onehot = batch["u_onehot"]
        mask = 1 - batch["padded"].float()  # 用来把那些填充的经验的TD-error置0，从而不让它们影响到学习（就是某些episode长度不一致，将短的后面填充0）
        if self.args.cuda:
            s = s.cuda()
            u = u.cuda()    # (n_episodes, max_episode_len, n_agents, 1)
            u_onehot = u_onehot.cuda()
            r = r.cuda()    # (n_episodes, max_episode_len, 1)
            s_next = s_next.cuda()
            mask = mask.cuda()  # (n_episodes, max_episode_len, 1)
            terminated = terminated.cuda()  # (n_episodes, max_episode_len, 1)



        # get eval q values(current q)
        self.eval_net.init_hidden(episode_num)
        q_evals, _ = self.eval_net.get_current_q_values(batch, self.max_episode_len)    # q_eval(n_episodes, max_episode_len, n_agents, n_actions)

        # pick the q values for the actions taken by each agent
        q_evals_chosen = th.gather(q_evals, dim=3, index=u).squeeze(3)    # q_eval(n_episodes, max_episode_len, n_agents)

        # get target q values(ground truth, next q)
        self.target_net.init_hidden(episode_num)
        q_targets, _ = self.target_net.get_next_q_values(batch, self.max_episode_len)     # q_targets(n_episodes, n_agents, n_actions)
        q_targets[avail_u_next == 0.0] = - 9999999  # 将不可用action的q值置为-9999999

        # pick the max q values
        if self.args.double_q:
            # if double_q, use eval_net based on s_next to determine the actions, and use the target net based on the actions to determine the q target values
            q_evals_next, _ = self.eval_net.get_next_q_values(batch, self.max_episode_len)
            q_evals_next = q_evals_next.detach()    # 要去除梯度
            q_evals_next[avail_u_next == 0] = -9999999
            cur_max_actions = th.argmax(q_evals_next, dim=3, keepdim=True)   # (n_episodes, max_episode_len, n_agents, 1)利用eval net选出最大next动作
            q_targets_chosen = th.gather(q_targets, dim=3, index=cur_max_actions).squeeze(3)
        else:
            # else, Use target_net based on s_next to determine the actions and q target values
            q_targets_chosen = th.max(q_targets, dim=3)[0]     # q_targets(n_episodes, max_episode_len, n_agents)

        # use mixer net to get q_tot & q_tot_target
        if self.args.alg == 'qplex':    # QPLEX
            # calculate v_tot
            v_tot = self.mixer(q_evals_chosen, s, is_v=True)

            # calculate a_tot
            q_evals_detached = q_evals.clone().detach()  # 提取数据不带梯度
            q_evals_detached[avail_u == 0] = -9999999  # 不能执行的动作赋值为负无穷
            max_action_qvals, max_action_index = q_evals_detached.max(dim=3)  # 最大的动作值及其索引

            max_action_index = max_action_index.detach().unsqueeze(3)  # (n_episodes, max_episode_len, n_agents, 1)去掉梯度
            is_max_action = (max_action_index == u).int().float()  # 0,1的矩阵，是否为最大动作 TODO

            a_tot = self.mixer(q_evals_chosen, s, actions=u_onehot, max_q_i=max_action_qvals, is_v=False)  # 计算优势值A

            # calculate q_tot
            q_tot = v_tot + a_tot

            # 因为QPLEX需要action_onehot作为输入，所以double_q时，要获取其选择动作的one_hot(eval net得到的action)
            if self.args.double_q:
                # get the eval_net max next action one hot (if double_q, Use eval_net based on s_next to determine the action)
                cur_max_actions_onehot = th.zeros_like(u_onehot)  # (n_episodes, max_episode_len, n_agents, n_actions)
                if self.args.cuda:
                    cur_max_actions_onehot = cur_max_actions_onehot.cuda()
                cur_max_actions_onehot = cur_max_actions_onehot.scatter_(3, cur_max_actions, 1)     # 将cur_max_actions位置直1变成one hot actions

                # calculate q_tot_target
                v_tot_target = self.target_mixer(q_targets_chosen, s_next, is_v=True)  # 计算状态价值V
                # TODO begin max_q_i=q_targets_chosen或者q_targets_max
                ## 如果是double_q，因为double_q时q_targets_chosen就是取的max，但是这里选择V值时是q_targets_max而不是q_targets_chosen（为什么？）
                ## 因为：q_targets_chosen并不是取的真的max，而是eval net的max action的q_targets，真正的q_targets是它本身取max
                q_targets_max = th.max(q_targets, dim=3)[0]
                a_tot_target = self.target_mixer(q_targets_chosen, s_next, actions=cur_max_actions_onehot, max_q_i=q_targets_max, is_v=False)  # 计算优势值A
                # TODO end

                q_tot_target = v_tot_target + a_tot_target  # 动作价值Q

            else:
                # 因为不用double_q时，q_targets_chosen已经是取的max，因为max(q) = v, 所以此时q_tot = v_tot
                q_tot_target = self.target_mixer(q_targets_chosen, s_next, is_v=True)  # 计算状态价值V（=Q值）

        else:   # VDN与QMIX
            q_tot = self.mixer(q_evals_chosen, s)     # q_tot(n_episodes, max_episode_len, 1)
            q_tot_target = self.target_mixer(q_targets_chosen, s_next)     # (n_episodes, max_episode_len, 1)

        # calculate td error & loss and train
        targets = r + self.gamma * q_tot_target * (1 - terminated)
        td_error = targets.detach() - q_tot     # 去掉target的梯度
        masked_td_error = mask * td_error  # 抹掉填充的经验的td_error(不能直接对masked_td_error求均值，因为有些被mask为0，所以不能算在里面)
        loss = (masked_td_error ** 2).sum() / mask.sum()
        # print('Loss is ', loss)
        self.optimizer.zero_grad()
        loss.backward()     # TODO separate报错，原因可能是调用多次forward而只调用一次backward，这两者必须交替执行.原因在于多个agent的rnn是并行的
        th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimizer.step()

        # update target net
        if train_step > 0 and train_step % self.args.target_update_cycle == 0:
            self._update_targets()

        return loss.item()  # 返回损失值

    def _update_targets(self):
        self.target_net.load_state(self.eval_net)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())

    def cuda(self):
        self.eval_net.cuda()
        self.target_net.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, train_step):
        num = str(train_step // self.args.save_cycle)   # 第几次保存的
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.eval_net.save_models(self.model_dir + '/' + num + '_rnn_net_params.pkl')
        th.save(self.mixer.state_dict(),  self.model_dir + '/' + num + '_mixer_net_params.pkl')

    def load_models(self):
        if os.path.exists(self.model_dir + '/rnn_net_params.pkl'):
            path_rnn = self.model_dir + '/rnn_net_params.pkl'
            path_vdn = self.model_dir + '/mixer_net_params.pkl'
            map_location = 'cuda:0' if self.args.cuda else 'cpu'
            self.eval_net.load_models(path_rnn)
            self.mixer.load_state_dict(th.load(path_vdn, map_location=map_location))
            print('Successfully load the model: {} and {}'.format(path_rnn, path_vdn))
        else:
            raise Exception("No model!")

    def get_q_and_q_tot_table(self):
        ''' 只用于求每种动作的q_table '''
        with th.no_grad():
            one_transition = {
                'o': np.array([[[[1.], [1.]]]]),      # (1,1,2,1)   n_episodes=1, max_len=1
                's': np.array([[[1.]]]),              # (1,1,1)
                'o_next': np.array([[[[1.], [1.]]]]),
                'u_onehot': np.array([[[[0., 0., 0.], [0., 0., 0.]]]]),  # (1,1,2,3)
                'avail_u': np.ones([1, 1, 2, 3], dtype=np.float)    # (1,1,2,3)
            }

            # batch to tensor
            for key in one_transition.keys():  # 把batch里的数据转化成tensor
                if key == 'u':
                    one_transition[key] = th.tensor(one_transition[key], dtype=th.long)  # action dtype is long
                else:
                    one_transition[key] = th.tensor(one_transition[key], dtype=th.float32)

            self.eval_net.init_hidden(episode_num=1)
            q_values, _ = self.eval_net.get_current_q_values(one_transition, max_episode_len=1)     # (1,1,2,3)
            q_table_i = q_values[0, 0, 0].cpu().numpy()
            q_table_j = q_values[0, 0, 1].cpu().numpy()
            q_tot_table = np.zeros((3, 3))
            if self.args.cuda:
                one_transition['s'] = one_transition['s'].cuda()
            for i in range(3):  # agent 0 action
                for j in range(3):  # agent 1 action
                    q_values_chosen = th.stack((q_values[:, :, 0, i], q_values[:, :, 1, j]), dim=1).unsqueeze(0).squeeze(3)  # (1,1,2)
                    if self.args.alg == 'qplex':
                        # calculate v_tot
                        v_tot = self.mixer(q_values_chosen, one_transition['s'], is_v=True)

                        # calculate a_tot
                        q_values_detached = q_values.clone()
                        q_values_detached[one_transition['avail_u'] == 0] = -9999999  # 不能执行的动作赋值为负无穷
                        max_action_qvals = q_values_detached.max(dim=3)[0]  # 最大的q值

                        # 所选动作的one hot
                        chosen_actions_one_hot = th.zeros_like(q_values)
                        if self.args.cuda:
                            chosen_actions_one_hot = chosen_actions_one_hot.cuda()
                        chosen_actions_one_hot[0, 0, 0, i] = 1  # agent 0
                        chosen_actions_one_hot[0, 0, 1, j] = 1  # agent 1


                        a_tot = self.mixer(q_values_chosen, one_transition['s'], actions=chosen_actions_one_hot, max_q_i=max_action_qvals, is_v=False)  # 计算优势值A
                        # calculate q_tot
                        q_tot_table[i, j] = v_tot + a_tot
                    else: # VDN & QMIX
                        q_tot_table[i, j] = self.mixer(q_values_chosen, one_transition['s']).item()

            return q_tot_table, q_table_i, q_table_j



