import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch as th

class RTWAgent(nn.Module):
    # Because all the agents share the same network, input_shape=obs_shape+n_actions+n_agents
    def __init__(self, input_shape, args):
        super(RTWAgent, self).__init__()
        self.args = args

        # RNN Net
        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)  # 只用的一个cell
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

        # Teammate Net
        '''  
            input: 
                h + dj (episode_num * n_agents * n_agents, rnn_hidden_dim + n_agents)
            output: 
                actions (episode_num * n_agents * n_agents, n_actions)   之后会reshape成actions (episode_num * n_agents, n_agents, n_actions)
         
        '''
        self.teammate_net = nn.Sequential(nn.Linear(args.rnn_hidden_dim + args.n_agents, args.hidden_dim),
                                          nn.ReLU(),
                                          nn.Linear(args.hidden_dim, args.n_actions)
                                          )

        # World Net
        '''  
            input: 
                obs + n_agents * n_actions (episode_num * n_agents, obs_shape + n_agents * n_actions)  即当前obs与其他所有agent的action(自己的action置为0)
            output: 
                obs_next (episode_num * n_agents, obs_shape)        即下一时刻的obs_next
        '''
        self.world_net = nn.Sequential(nn.Linear(args.obs_shape + args.n_agents * args.n_actions, args.hidden_dim),
                                       nn.ReLU(),
                                       nn.Linear(args.hidden_dim, args.obs_shape)
                                       )

        # Reflecton Net
        '''  
            input: 
                obs (episode_num * n_agents, obs_shape) 
                obs_next (episode_num * n_agents, obs_shape)            即World Net的输出
                h_repeat (episode_num * n_agents, n,agents, rnn_hidden_dim)    要先将h repeat能和actions拼接
                actions (episode_num * n_agents, n_agents, n_actions)   即Teammate Net的输出
             output: 
                obs_next (episode_num * n_agents, obs_shape)
        '''
        self.w_q = nn.Linear(args.obs_shape + args.obs_shape, args.attn_dim)
        self.w_k = nn.Linear(args.n_actions, args.attn_dim)
        self.w_v = nn.Sequential(nn.Linear(args.rnn_hidden_dim + args.n_actions, args.attn_dim),
                                 nn.ReLU(),
                                 nn.Linear(args.attn_dim, args.n_actions)
                                 )


    def forward(self, inputs, hidden_state, obs, obs_next, u, avail_u, target=False, test_mode=False, agent_num=0):
        # inputs(episode_num * n_agents, obs + num_actions + n_agents)
        episode_num = int(inputs.size(0) / self.args.n_agents)

        # 得到q值
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)

        if test_mode:   # 若是测试或者采集数据时，只有一个agent作为batch，bs = 1，则Teammates的bs = 1 * n_agents
            dj = th.eye(self.args.n_agents).unsqueeze(0).view(-1, self.args.n_agents, self.args.n_agents).cuda()
            h_repeat = h.repeat(1, self.args.n_agents).view(-1, self.args.n_agents, self.args.rnn_hidden_dim)
            inps_t = th.cat([h_repeat, dj], dim=-1).view(-1, self.args.rnn_hidden_dim + self.args.n_agents)
            mask = th.ones(1 * self.args.n_agents, 1).cuda()
            if self.args.not_self_model:
                inps_t[agent_num] = 0   # 不建模自己的action
                mask[agent_num] = 0

            actions = self.teammate_net(inps_t).view(-1, self.args.n_agents, self.args.n_actions)
            mask = mask.view(1, self.args.n_agents, 1)

            actions_clone = actions.clone()

            actions_clone[avail_u == 0.0] = -1e9
            actions_clone = actions_clone.argmax(dim=-1, keepdim=True)
            actions_onehot = th.zeros((1, self.args.n_agents, self.args.n_actions))
            actions_onehot = actions_onehot.scatter(-1, actions_clone[:].cpu(), 1).cuda()


            # world net
            actions_onehot = th.zeros((1, self.args.n_agents, self.args.n_actions))
            actions_onehot = actions_onehot.scatter(-1, actions_clone[:].cpu(), 1).cuda()

            if self.args.not_self_model:
                actions_onehot = actions_onehot * mask
            actions_onehot = actions_onehot.view(-1, self.args.n_agents * self.args.n_actions)
            inps_w = th.cat([obs, actions_onehot], dim=-1)
            obs_next_hat = self.world_net(inps_w)

            # reflection net
            inps_q = th.cat([obs, obs_next_hat], dim=-1)
            query = self.w_q(inps_q).unsqueeze(1)

            actions_onehot = actions_onehot.view(-1, self.args.n_actions)

            key = self.w_k(actions_onehot).view(-1, self.args.n_agents, self.args.attn_dim).transpose(1, 2)

            inps_v = th.cat([h_repeat.view(-1, self.args.rnn_hidden_dim), actions_onehot], dim=-1)
            value = self.w_v(inps_v).view(-1, self.args.n_agents, self.args.n_actions)
            attn_score = th.bmm(query / (self.args.attn_dim ** (1 / 2)), key)

            if self.args.not_self_model:
                attn_score[:, :, agent_num] = -1e9

            attn_score = F.softmax(attn_score, dim=-1).reshape(-1, self.args.n_agents, 1)
            q_r = value * attn_score
            q_r = th.sum(q_r, dim=1).view(-1, self.args.n_actions)
            q += q_r
            return q, h

        else:   # 训练时，由于所有agent共享一网络，bs = episode_num * n_agents。那么Teammate Net的 bs = episode_num * n_agents * n_agents
            h_repeat = h.view(-1, self.args.n_agents, self.args.rnn_hidden_dim).repeat(1, self.args.n_agents, 1).view(-1,
                                                                                                                  self.args.n_agents,
                                                                                                                  self.args.n_agents,
                                                                                                                  self.args.rnn_hidden_dim)
            dj = th.eye(self.args.n_agents).unsqueeze(0).expand(episode_num, -1, -1).repeat(1, self.args.n_agents,
                                                                                            1).view(-1,
                                                                                                    self.args.n_agents,
                                                                                                    self.args.n_agents,
                                                                                                    self.args.n_agents).cuda()
            inps_t = th.cat([h_repeat, dj], dim=-1)
            mask = th.ones(episode_num, self.args.n_agents, self.args.n_agents, 1)
            if self.args.not_self_model:
                for i in range(self.args.n_agents):
                    inps_t[:, i, i] = 0
                    mask[:, i, i] = 0
            inps_t = inps_t.view(-1, self.args.rnn_hidden_dim + self.args.n_agents)
            actions = self.teammate_net(inps_t.detach()).view(-1, self.args.n_agents, self.args.n_agents, self.args.n_actions)
            mask = mask.view(-1, self.args.n_agents, self.args.n_agents, 1).cuda()

            if target:
                loss_t = 0
            else:
                actions_label = u.view(-1, self.args.n_agents).repeat(1, self.args.n_agents).view(-1, 1).squeeze(-1)
                # loss_t = self.calc_teammatenet_loss(actions.view(-1, self.args.n_actions), actions_label.detach(), mask.view(-1, 1))
                loss_t = 0

            actions_clone = actions.clone()

            avail_u = avail_u.view(-1, self.args.n_agents, self.args.n_actions).repeat(1, self.args.n_agents, 1).view(-1,
                                                                                                                  self.args.n_agents,
                                                                                                                  self.args.n_agents,
                                                                                                                  self.args.n_actions)
            actions_clone[avail_u == 0.0] = -1e9
            actions_clone = actions_clone.argmax(dim=-1, keepdim=True)
            actions_onehot = th.zeros((episode_num, self.args.n_agents, self.args.n_agents, self.args.n_actions))
            actions_onehot = actions_onehot.scatter(-1, actions_clone[:].cpu(), 1).cuda()

            # world net
            if self.args.not_self_model:
                actions_onehot = actions_onehot * mask
            actions_onehot = actions_onehot.view(-1, self.args.n_agents * self.args.n_actions)
            inps_w = th.cat([obs, actions_onehot], dim=-1)
            obs_next_hat = self.world_net(inps_w.detach())

            if target:
                loss_w = 0
            else:
                # loss_w = self.calc_worldnet_loss(obs_next_hat, obs_next.detach())
                loss_w = 0

            # reflection net
            # inps_q = th.cat([obs, obs_next_hat], dim=-1)
            # query = self.w_q(inps_q.detach()).unsqueeze(1)
            #
            # actions_onehot = actions_onehot.view(-1, self.args.n_actions)

            inps_q = th.cat([obs, obs_next], dim=-1)
            query = self.w_q(inps_q.detach()).unsqueeze(1)

            actions_onehot = th.zeros((episode_num, self.args.n_agents, self.args.n_agents, self.args.n_actions))
            u_repeat = u.view(-1, self.args.n_agents, 1).repeat(1, self.args.n_agents, 1).view(-1, self.args.n_agents, self.args.n_agents, 1)
            actions_onehot = actions_onehot.scatter(-1, u_repeat[:].cpu(), 1).cuda()
            if self.args.not_self_model:
                actions_onehot = actions_onehot * mask

            actions_onehot = actions_onehot.view(-1, self.args.n_actions)

            key = self.w_k(actions_onehot.detach()).view(-1, self.args.n_agents, self.args.attn_dim).transpose(1, 2)

            inps_v = th.cat([h_repeat.view(-1, self.args.rnn_hidden_dim), actions_onehot], dim=-1)
            value = self.w_v(inps_v.detach()).view(-1, self.args.n_agents, self.args.n_actions)
            attn_score = th.bmm(query / (self.args.attn_dim ** (1 / 2)), key).view(-1, self.args.n_agents, self.args.n_agents)

            if self.args.not_self_model:
                for i in range(self.args.n_agents):
                    attn_score[:, i, i] = -1e9

            attn_score = F.softmax(attn_score, dim=-1).reshape(-1, self.args.n_agents, 1)
            q_r = value * attn_score
            q_r = th.sum(q_r, dim=1).view(-1, self.args.n_actions)
            q += q_r
            return q, h, loss_t, loss_w

    def calc_teammatenet_loss(self, action_j_hat, action_j, mask):
        debug_info = F.softmax(action_j_hat, dim=-1).argmax(dim=-1)
        teammate_loss = F.cross_entropy(action_j_hat, action_j, reduction='none')
        return (teammate_loss * mask).mean() * self.args.teammate_loss_weight

    def calc_worldnet_loss(self, o_next_hat, o_next):
        world_loss = ((o_next_hat - o_next) ** 2).mean()
        return world_loss * self.args.world_loss_weight



if __name__ == '__main__':
    from torchsummary import summary
    from runner import Runner
    from smac.env import StarCraft2Env
    from common.arguments import get_common_args, get_coma_args, get_RTW_args, get_mixer_args, get_centralv_args, \
        get_reinforce_args, get_commnet_args, get_g2anet_args
    from env.single_state_matrix_game import TwoAgentsMatrixGame
    from torchviz import make_dot

    args = get_common_args()
    get_mixer_args(args)
    get_RTW_args(args)
    if args.env == 'smac':
        env = StarCraft2Env(map_name=args.map,
                            step_mul=args.step_mul,
                            difficulty=args.difficulty,
                            game_version=args.game_version,
                            replay_dir=args.replay_dir)
    else:
        raise ValueError("env not found")

    env_info = env.get_env_info()  # env info for other args
    args.n_actions = env_info["n_actions"]
    args.n_agents = env_info["n_agents"]
    args.state_shape = env_info["state_shape"]
    args.obs_shape = env_info["obs_shape"]
    args.episode_limit = env_info["episode_limit"]
    episode_num = 1


    model = RTWAgent(input_shape= args.obs_shape + args.n_actions + args.n_agents, args=args).cuda()



    test_mode = False
    args.not_self_model = True
    if test_mode:

        avail_actions = [[0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0], [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                         [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0], [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                         [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]]

        avail_actions = th.tensor(avail_actions, dtype=th.long).unsqueeze(0).cuda()

        inputs = th.randn((1, args.obs_shape + args.n_actions + args.n_agents)).cuda()
        h = th.randn((1, args.rnn_hidden_dim)).cuda()
        obs = th.randn((1, args.obs_shape)).cuda()
        model(inputs, h, obs, None, None, avail_actions, test_mode=test_mode)
    else:
        inputs = th.randn((episode_num * args.n_agents, args.obs_shape + args.n_actions + args.n_agents)).cuda()
        h = th.randn((episode_num * args.n_agents, args.rnn_hidden_dim)).cuda()
        obs = th.randn((episode_num * args.n_agents, args.obs_shape)).cuda()
        obs_next = th.randn((episode_num * args.n_agents, args.obs_shape)).cuda()

        actions = [1,2,3,4,5]
        # for agent_id in range(args.n_agents):
        #     avail_actions = env.get_avail_agent_actions(agent_id)
        #     avail_actions_ind = np.nonzero(avail_actions)[0]
        #     action = np.random.choice(avail_actions_ind)
        #     actions.append(action)
        #
        # avail_actions = env.get_avail_actions()
        avail_actions = [[0,1,1,1,1,1,0,0,0,0,0], [0,1,1,1,1,1,0,0,0,0,0], [0,1,1,1,1,1,0,0,0,0,0], [0,1,1,1,1,1,0,0,0,0,0], [0,1,1,1,1,1,0,0,0,0,0]]
        avail_actions = th.tensor(avail_actions, dtype=th.long).cuda()  # (5, 11)

        u = th.tensor(actions, dtype=th.long).cuda()    # (5,)
        q, h, loss_t, loss_w = model(inputs, h, obs, obs_next, u, avail_actions, test_mode=test_mode)
        loss_td = ((q - (q/2).detach()) ** 2).mean()
        loss = loss_td + loss_t + loss_w

        # graph_all = make_dot(loss)
        # graph_t = make_dot(loss_t)
        # graph_w = make_dot(loss_w)
        # graph_td = make_dot(loss_td)
        #
        # graph_all.view('loss_all', '.\\figure\\')
        # graph_t.view('loss_t', '.\\figure\\')
        # graph_w.view('loss_w', '.\\figure\\')
        # graph_td.view('loss_td', '.\\figure\\')

