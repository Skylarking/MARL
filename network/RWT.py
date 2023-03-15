import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch as th

# 预测另一个agent的action
class TeammateModel(nn.Module):
    def __init__(self, args):
        super(TeammateModel, self).__init__()
        self.args = args

        # net
        self.teammate_net = nn.Sequential(nn.Linear(args.rnn_hidden_dim + args.n_agents, args.hidden_dim),
                                          nn.ReLU(),
                                          nn.Linear(args.hidden_dim, args.n_actions)
                                          )
    def forward(self, h, dj):
        inp = th.cat([h, dj], dim=-1)
        action_j = self.teammate_net(inp)
        return action_j



# 预测环境下一个时刻的信息
class WorldModel(nn.Module):
    def __init__(self, args):
        super(WorldModel, self).__init__()
        self.args = args

        # net
        self.world_net = nn.Sequential(nn.Linear(args.obs_shape + args.n_agents * args.n_actions, args.hidden_dim),
                                       nn.ReLU(),
                                       nn.Linear(args.hidden_dim, args.obs_shape)
                                       )

    def forward(self, o, actions):
        # (n_episodes, max_episode_len, n_agents, obs_shape/n_actions)
        inps = th.cat([o, actions], dim=-1)
        o_next = self.world_net(inps)
        return o_next

class ReflectionModel(nn.Module):
    def __init__(self, args):
        super(ReflectionModel, self).__init__()

        self.args = args

        self.num_kernel = args.num_kernel   # 注意力头数

        self.key_extractors = nn.ModuleList()
        self.agents_extractors = nn.ModuleList()
        self.action_extractors = nn.ModuleList()

        # net
        self.w_q = nn.Linear(args.obs_shape + args.obs_shape, args.attn_dim)
        self.w_k = nn.Linear(args.n_actions, args.attn_dim)

        self.w_v = nn.Sequential(nn.Linear(args.rnn_hidden_dim + args.n_actions, args.attn_dim),
                                 nn.ReLU(),
                                 nn.Linear(args.attn_dim, args.n_actions)
                                 )

    def forward(self, o, o_next, h, actions, test_mode=False, agent_num=0):
        # o,o_next(bs, obs_shape)
        # h(bs, rnn_hidden_dim)
        # actions(bs, n_agents, n_actions)
        query = self.w_q(th.cat([o, o_next], dim=-1)).unsqueeze(1)
        key = self.w_k(actions.view(-1, self.args.n_actions)).view(-1, self.args.n_agents, self.args.attn_dim).transpose(1, 2)


        if test_mode:
            h_repeat = h.repeat(1, self.args.n_agents).view(-1, self.args.rnn_hidden_dim)
            inps_v = th.cat([h_repeat, actions.view(-1, self.args.n_actions)], dim=-1)
            value = self.w_v(inps_v).view(-1, self.args.n_agents, self.args.n_actions)
            attn_score = th.bmm(query / (self.args.attn_dim ** (1 / 2)), key)
            attn_score = attn_score.view(-1, self.args.n_agents)
            attn_score[:, agent_num] = -1e9
            attn_score = F.softmax(attn_score, dim=-1).reshape(-1, self.args.n_agents, 1)
            q = value * attn_score
            q = th.sum(q, dim=1)
            return q.view(-1, self.args.n_actions)

        else:
            h_repeat = h.view(-1, self.args.n_agents, self.args.rnn_hidden_dim).repeat(1, self.args.n_agents, 1).view(
                -1, self.args.rnn_hidden_dim)

            inps_v = th.cat([h_repeat, actions.view(-1, self.args.n_actions)], dim=-1)
            value = self.w_v(inps_v).view(-1, self.args.n_agents, self.args.n_actions)
            attn_score = th.bmm(query / (self.args.attn_dim ** (1 / 2)), key)
            attn_score = attn_score.view(-1, self.args.n_agents, self.args.n_agents)
            for i in range(self.args.n_agents):
                attn_score[:, i, i] = -1e9  # 置负无穷，即本身的建模得到的分数为0
            attn_score = F.softmax(attn_score, dim=-1).reshape(-1, self.args.n_agents, 1)
            q = value * attn_score
            q = th.sum(q, dim=1)
            return q.view(-1, self.args.n_actions)

class RTWAgent(nn.Module):
    # Because all the agents share the same network, input_shape=obs_shape+n_actions+n_agents
    def __init__(self, input_shape, args):
        super(RTWAgent, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim) # 只用的一个cell
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

        self.teammate_net = TeammateModel(args)
        self.world_net = WorldModel(args)
        self.reflection_net = ReflectionModel(args)

    def forward(self, inputs, hidden_state, obs, obs_next, u, target=False, test_mode=False, agent_num=0):
        # inputs(episode_num * n_agents, obs + num_actions + n_agents)
        episode_num = int(inputs.size(0) / self.args.n_agents)
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)

        if test_mode:
            h_repeat = h.detach().repeat(1, self.args.n_agents).view(self.args.n_agents, self.args.rnn_hidden_dim)
            dj = th.eye(self.args.n_agents).unsqueeze(0).view(self.args.n_agents, self.args.n_agents).cuda()
            h_repeat[agent_num] = 0
            dj[agent_num] = 0
            actions = self.teammate_net(h_repeat, dj)
            actions[agent_num] = 0
            actions_masked = actions
            o_next_hat = self.world_net(obs, actions_masked.view(-1, self.args.n_agents * self.args.n_actions))
            q_reflect = self.reflection_net(obs, o_next_hat.detach(), h, actions_masked.detach().view(-1, self.args.n_actions), test_mode=True, agent_num=agent_num)
            q += q_reflect

            return q_reflect, h


        # Teammate model, get the actions of agent_-i
        # h_repeat(ep_num ,n_agents, n_agents, rnn_hidden_dim) # dj(ep_num ,n_agents, n_agents, n_agents)
        h_repeat = h.detach().view(episode_num, self.args.n_agents, -1).repeat(1, self.args.n_agents, 1).view(-1, self.args.n_agents, self.args.n_agents, self.args.rnn_hidden_dim)
        dj = th.eye(self.args.n_agents).unsqueeze(0).expand(episode_num, -1, -1).repeat(1, self.args.n_agents, 1).view(-1, self.args.n_agents, self.args.n_agents, self.args.n_agents).cuda()

        # 将自身agent的输入置为0，因为只建模其他agent
        for agent_id in range(self.args.n_agents):
            h_repeat[:, agent_id, agent_id] = 0
            dj[:, agent_id, agent_id] = 0

        h_repeat = h_repeat.view(-1, self.args.rnn_hidden_dim)
        dj = dj.view(-1, self.args.n_agents)


        actions = self.teammate_net(h_repeat, dj)
        actions1 = actions.view(episode_num, self.args.n_agents, self.args.n_agents, self.args.n_actions)
        mask = th.ones([episode_num, self.args.n_agents, self.args.n_agents]).cuda()
        mask1 = th.ones([episode_num, self.args.n_agents, self.args.n_agents, self.args.n_actions]).cuda()
        # u = u.view([episode_num, self.args.n_agents, self.args.n_actions])
        # 自己action不用建模，要把自己action mask掉
        for agent_id in range(self.args.n_agents):
            # actions[:, agent_id, agent_id] = u[:, agent_id]
            mask[:, agent_id, agent_id] = 0
            mask1[:, agent_id, agent_id] = 0
        actions_masked = (actions1 * mask1).view(-1, self.args.n_agents, self.args.n_actions)


        if target:
            loss1 = None
        else:
            actions_label = u.view(episode_num, self.args.n_agents, -1).repeat(1, self.args.n_agents, 1).view(-1, 1).squeeze(-1)
            loss1 = self.calc_teammatenet_loss(actions, actions_label, mask.view(-1, 1).squeeze(-1))


        # world model
        o_next_hat = self.world_net(obs, actions_masked.detach().squeeze(-1).view(-1, self.args.n_agents * self.args.n_actions))
        if target:
            loss2 = None
        else:
            loss2 = self.calc_worldnet_loss(o_next_hat, obs_next)

        # reflection model
        q_reflect = self.reflection_net(obs, o_next_hat.detach(), h, actions_masked.detach().view(-1, self.args.n_actions))

        q += q_reflect

        return q_reflect, h, loss1, loss2

    def calc_teammatenet_loss(self, action_j_hat, action_j, mask):
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
    summary(model=model, input_size=[
        (args.obs_shape + args.n_actions + args.n_agents,),
        (args.rnn_hidden_dim,),
        (args.obs_shape,),
        (args.obs_shape,),
        (1,)], batch_size=episode_num * args.n_agents, device="cuda")
    # summary(model=model, input_size=[
    #             (1 * args.n_agents, args.obs_shape + args.n_actions + args.n_agents),
    #             (1 * args.n_agents, args.rnn_hidden_dim),
    #             (1 * args.n_agents, args.obs_shape),
    #             (1 * args.n_agents, args.obs_shape),
    #             (1 * args.n_agents, 1)], device="cuda")
    # inputs, hidden_state, obs, obs_next, u, target=False, test_mode=False, agent_num=0