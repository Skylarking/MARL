import numpy as np
import torch.nn as nn
import torch.nn.functional as f
import torch as th
from network.mixer import QtranQBase, QtranQAlt, QtranV, QMixMixer
import copy
import os


class QTRANLearner():
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

        if args.alg == 'qtran_base':
            self.mixer = QtranQBase(args)
        elif args.alg == 'qtran_alt':
            self.mixer = QtranQAlt(args)
        else:
            raise ValueError("Mixer {} not recognised.".format(args.alg))

        self.target_mixer = copy.deepcopy(self.mixer)
        self.params += list(self.mixer.parameters())

        self.v = QtranV(args)
        self.params += list(self.v.parameters())

        self.q_sum_mixer = QMixMixer(args)
        self.params += list(self.q_sum_mixer.parameters())

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
        mask = 1 - batch["padded"].float().squeeze(-1)  # 用来把那些填充的经验的TD-error置0，从而不让它们影响到学习（就是某些episode长度不一致，将短的后面填充0）
        if self.args.cuda:
            s = s.cuda()
            u = u.cuda()    # (n_episodes, max_episode_len, n_agents, 1)
            r = r.cuda()    # (n_episodes, max_episode_len, 1)
            s_next = s_next.cuda()
            mask = mask.cuda()  # (n_episodes, max_episode_len, 1)
            terminated = terminated.cuda()  # (n_episodes, max_episode_len, 1)


        # get eval q values(current q)
        self.eval_net.init_hidden(episode_num)
        individual_q_evals, hidden_evals = self.eval_net.get_current_q_values(batch, self.max_episode_len)    # q_eval(n_episodes, max_episode_len, n_agents, n_actions)

        # get target q values(ground truth, next q)
        self.target_net.init_hidden(episode_num)
        individual_q_targets, hidden_targets = self.target_net.get_next_q_values(batch, self.max_episode_len)     # q_targets(n_episodes, max_episode_len, n_agents, n_actions)

        # 得到当前时刻和下一时刻每个agent的局部最优动作及其one_hot表示（用于QtranV网络的输入得到q_tot_max）
        individual_q_clone = individual_q_evals.clone()
        individual_q_clone[avail_u == 0.0] = - 999999
        individual_q_targets[avail_u_next == 0.0] = - 9999999  # 将不可用action的q值置为-9999999

        opt_onehot_eval = th.zeros(*individual_q_clone.shape)
        opt_action_eval = individual_q_clone.argmax(dim=3, keepdim=True)
        opt_onehot_eval = opt_onehot_eval.scatter(-1, opt_action_eval[:, :].cpu(), 1)   # (n_episodes, n_agents, n_actions)  转换为one hot action

        opt_onehot_target = th.zeros(*individual_q_targets.shape)
        opt_action_target = individual_q_targets.argmax(dim=3, keepdim=True)
        opt_onehot_target = opt_onehot_target.scatter(-1, opt_action_target[:, :].cpu(), 1)

        # ---------------------------------------------L_td-------------------------------------------------------------
        # 这里的joint_q_evals是根据所选动作计算的q_tot，joint_q_targets是下一时刻根据target最优动作选择的q_tot，v和动作无关
        # joint_q、v的维度为(episode个数, max_episode_len, 1), 而且joint_q在后面的l_nopt还要用到
        joint_q_evals, joint_q_targets, v = self.get_qtran(batch, hidden_evals, hidden_targets, opt_onehot_target)

        y_dqn = r.squeeze(-1) + self.args.gamma * joint_q_targets * (1 - terminated.squeeze(-1))
        td_error = joint_q_evals - y_dqn.detach()
        l_td = ((td_error * mask) ** 2).sum() / mask.sum()
        # ---------------------------------------------L_td-------------------------------------------------------------

        # ---------------------------------------------L_opt------------------------------------------------------------
        # 将所有agent局部最优动作的Q值相加
        # 这里要使用individual_q_clone，它把不能执行的动作Q值改变了，使用individual_q_evals可能会使用不能执行的动作的Q值
        q_sum_opt = individual_q_clone.max(dim=-1)[0].sum(dim=-1)  # (episode个数, max_episode_len) # TODO 原始QTRAN
        # TODO 原始QTRAN是直接greedy Q值相加(即VDN的方式，这里换成QMIX的方式)
        # q_sum_opt = individual_q_clone.max(dim=-1)[0]  # (episode个数, max_episode_len, n_agents)
        # q_sum_opt = self.q_sum_mixer(q_sum_opt, s).squeeze(-1) # (episode个数, max_episode_len)

        # 重新得到joint_q_hat_opt是根据eval最优动作计算的（它和joint_q_evals的区别是前者输入的动作是局部最优动作，后者输入的动作是执行的动作）
        # (episode个数, max_episode_len)
        joint_q_hat_opt, _, _ = self.get_qtran(batch, hidden_evals, hidden_targets, opt_onehot_eval, hat=True)
        opt_error = q_sum_opt - joint_q_hat_opt.detach() + v  # 计算l_opt时需要将joint_q_hat_opt固定
        l_opt = ((opt_error * mask) ** 2).sum() / mask.sum()
        # ---------------------------------------------L_opt------------------------------------------------------------

        # ---------------------------------------------L_nopt-----------------------------------------------------------
        # 每个agent的执行动作的Q值,(episode个数, max_episode_len, n_agents, 1)
        q_individual = th.gather(individual_q_evals, dim=-1, index=u).squeeze(-1)
        q_sum_nopt = q_individual.sum(dim=-1)  # (episode个数, max_episode_len)

        nopt_error = q_sum_nopt - joint_q_evals.detach() + v  # 计算l_nopt时需要将joint_q_evals固定
        nopt_error = nopt_error.clamp(max=0)
        l_nopt = ((nopt_error * mask) ** 2).sum() / mask.sum()
        # ---------------------------------------------L_nopt-----------------------------------------------------------

        # total loss
        loss = l_td + self.args.lambda_opt * l_opt + self.args.lambda_nopt * l_nopt
        # loss = l_td + self.args.lambda_opt * l_opt
        self.optimizer.zero_grad()
        loss.backward()
        th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimizer.step()

        # update target net
        if train_step > 0 and train_step % self.args.target_update_cycle == 0:
            self._update_targets()

        return loss.item()  # 返回损失值

    def get_qtran(self, batch, hidden_evals, hidden_targets, local_opt_actions, hat=False):
        '''
        利用QtranV网络计v值,利用QTranMixer计算q_evals与q_targets

        参数：
            hat=True表示是不是eval网络，True表示是eval，False表示是target
        '''
        episode_num, max_episode_len, _, _ = hidden_targets.shape
        states = batch['s'][:, :max_episode_len]
        states_next = batch['s_next'][:, :max_episode_len]
        u_onehot = batch['u_onehot'][:, :max_episode_len]
        if self.args.cuda:
            states = states.cuda()
            states_next = states_next.cuda()
            u_onehot = u_onehot.cuda()
            hidden_evals = hidden_evals.cuda()
            hidden_targets = hidden_targets.cuda()
            local_opt_actions = local_opt_actions.cuda()
        if hat:
            # 神经网络输出的q_eval、q_target、v的维度为(episode_num * max_episode_len, 1)
            q_evals = self.mixer(states, hidden_evals, local_opt_actions)   # (episode_num * max_episode_len, 1)
            q_targets = None
            v = None

            # 把q_eval维度变回(episode_num, max_episode_len)
            q_evals = q_evals.view(episode_num, -1, 1).squeeze(-1)
        else:
            q_evals = self.mixer(states, hidden_evals, u_onehot)        # (episode_num * max_episode_len, 1)
            q_targets = self.target_mixer(states_next, hidden_targets, local_opt_actions)   # (episode_num * max_episode_len, 1)
            v = self.v(states, hidden_evals)    # (episode_num * max_episode_len, 1)
            # 把q_eval、q_target、v维度变回(episode_num, max_episode_len)
            q_evals = q_evals.view(episode_num, -1, 1).squeeze(-1)
            q_targets = q_targets.view(episode_num, -1, 1).squeeze(-1)
            v = v.view(episode_num, -1, 1).squeeze(-1)

        return q_evals, q_targets, v    # (episode_num, max_episode_len)

    def _update_targets(self):
        self.target_net.load_state(self.eval_net)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())

    def cuda(self):
        self.eval_net.cuda()
        self.target_net.cuda()
        self.v.cuda()
        self.q_sum_mixer.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, train_step):
        num = str(train_step // self.args.save_cycle)   # 第几次保存的
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.eval_net.save_models(self.model_dir + '/' + num + '_rnn_net_params.pkl')
        th.save(self.mixer.state_dict(),  self.model_dir + '/' + num + '_mixer_net_params.pkl')
        th.save(self.v.state_dict(), self.model_dir + '/' + num + '_v_net_params.pkl')

    def load_models(self):
        if os.path.exists(self.model_dir + '/rnn_net_params.pkl'):
            path_rnn = self.model_dir + '/rnn_net_params.pkl'
            path_qtran_mixer = self.model_dir + '/mixer_net_params.pkl'
            path_qtran_v = self.model_dir + '/v_net_params.pkl'
            map_location = 'cuda:0' if self.args.cuda else 'cpu'
            self.eval_net.load_models(path_rnn)
            self.mixer.load_state_dict(th.load(path_qtran_mixer, map_location=map_location))
            self.v.load_state_dict((th.load(path_qtran_v, map_location=map_location)))
            print('Successfully load the model: {} and {}'.format(path_rnn, path_qtran_mixer))
        else:
            raise Exception("No model!")

    def get_q_and_q_tot_table(self):
        ''' 只用于求每种动作的q_table '''
        with th.no_grad():
            one_transition = {
                'o': np.array([[[[1.], [1.]]]]),      # (1,1,2,1)   n_episodes=1, max_len=1
                's': np.array([[[1.]]]),              # (1,1,1)
                'o_next': np.array([[[[1.], [1.]]]]),
                'u_onehot': np.array([[[[0., 0., 0.], [0., 0., 0.]]]])  # (1,1,2,3)
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
            for i in range(3):
                for j in range(3):
                    q_evals = th.stack((q_values[:, :, 0, i], q_values[:, :, 1, j]), dim=1).unsqueeze(0).squeeze(3)  # (1,1,2)
                    hidden_evals = th.zeros([1, 1, self.args.n_agents, self.args.rnn_hidden_dim])
                    states = one_transition['s']
                    u_onehot = one_transition['u_onehot']
                    u_onehot[:, :, 0, i] = 1
                    u_onehot[:, :, 1, j] = 1
                    if self.args.cuda:
                        states = states.cuda()
                        hidden_evals = hidden_evals.cuda()
                        u_onehot = u_onehot.cuda()
                    q_tot_table[i, j] = self.mixer(states, hidden_evals, u_onehot).item()
            return q_tot_table, q_table_i, q_table_j