import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

###############################################################
####################      VDN Mixer   #########################
###############################################################
class VDNMixer(nn.Module):
    ''' VDN是直接将q_values相加 '''
    def __init__(self, args):
        super(VDNMixer, self).__init__()
        self.args = args

    def forward(self, q_values, states=None):
        return th.sum(q_values, dim=2, keepdim=True)

###############################################################
####################      QMix Mixer   ########################
###############################################################
class QMixMixer(nn.Module):
    '''
    用一个超网络生成q_values的weight(>0)和bias，有两个超网络生成两层参数(每一层参数是一个矩阵)
    超网络(2个)：
        输入：state
        输出：weights(>0)(先输出向量，再reshape成矩阵)，bias (bias本身就是一个向量)
    '''
    def __init__(self, args):
        super(QMixMixer, self).__init__()
        self.args = args
        # 因为生成的hyper_w1需要是一个矩阵，而pyth神经网络只能输出一个向量，
        # 所以就先输出长度为需要的 矩阵行*矩阵列 的向量，然后再转化成矩阵

        # args.n_agents是使用hyper_w1作为参数的网络的输入维度，args.qmix_hidden_dim是网络隐藏层参数个数
        # 从而经过hyper_w1得到(经验条数，args.n_agents * args.qmix_hidden_dim)的矩阵
        if args.two_hyper_layers:
            self.hyper_w1 = nn.Sequential(nn.Linear(args.state_shape, args.hyper_hidden_dim),
                                          nn.ReLU(),
                                          nn.Linear(args.hyper_hidden_dim, args.n_agents * args.qmix_hidden_dim))
            # 经过hyper_w2得到(经验条数, 1)的矩阵
            self.hyper_w2 = nn.Sequential(nn.Linear(args.state_shape, args.hyper_hidden_dim),
                                          nn.ReLU(),
                                          nn.Linear(args.hyper_hidden_dim, args.qmix_hidden_dim))
        else:
            self.hyper_w1 = nn.Linear(args.state_shape, args.n_agents * args.qmix_hidden_dim)
            # 经过hyper_w2得到(经验条数, 1)的矩阵
            self.hyper_w2 = nn.Linear(args.state_shape, args.qmix_hidden_dim * 1)

        # hyper_w1得到的(经验条数，args.qmix_hidden_dim)矩阵需要同样维度的hyper_b1
        self.hyper_b1 = nn.Linear(args.state_shape, args.qmix_hidden_dim)
        # hyper_w2得到的(经验条数，1)的矩阵需要同样维度的hyper_b1
        self.hyper_b2 = nn.Sequential(nn.Linear(args.state_shape, args.qmix_hidden_dim),
                                      nn.ReLU(),
                                      nn.Linear(args.qmix_hidden_dim, 1)
                                      )

    def forward(self, q_values, states):
        # states(episode_num, max_episode_len, state_shape)
        # q_values(episode_num, max_episode_len, n_agents)
        episode_num = q_values.size(0)  # n_episodes
        q_values = q_values.view(-1, 1, self.args.n_agents)  # (episode_num * max_episode_len, 1, n_agents) = (1920,1,5)
        states = states.reshape(-1, self.args.state_shape)  # (episode_num * max_episode_len, state_shape)

        w1 = th.abs(self.hyper_w1(states))  # (episode_num * max_episode_len, n_agents * qmix_hidden_dim) = (1920, 160)
        b1 = self.hyper_b1(states)  # (episode_num * max_episode_len, qmix_hidden_dim) = (1920, 32)

        w1 = w1.view(-1, self.args.n_agents, self.args.qmix_hidden_dim) # (episode_num * max_episode_len, n_agents, qmix_hidden_dim) = (1920, 5, 32)
        b1 = b1.view(-1, 1, self.args.qmix_hidden_dim)  # (1920, 1, 32)

        hidden = F.elu(th.bmm(q_values, w1) + b1)  # (1920, 1, 32)

        w2 = th.abs(self.hyper_w2(states))  # (1920, 32)
        b2 = self.hyper_b2(states)  # (1920, 1)

        w2 = w2.view(-1, self.args.qmix_hidden_dim, 1)  # (1920, 32, 1)
        b2 = b2.view(-1, 1, 1)  # (1920, 1， 1)

        q_tot = th.bmm(hidden, w2) + b2  # (1920, 1, 1)
        q_tot = q_tot.view(episode_num, -1, 1)  # (32, 60, 1)
        return q_tot    # (episode_num, max_episode_len, 1)

###############################################################
####################      QPLEX Mixer   #######################
###############################################################
class DMAQ_SI_Weight(nn.Module):
    '''
    计算λ_i的部分，即A_tot的参数

    参数:
        num_kernel: k的值，即有多少个注意力头

        embedding层，都是nn.ModuleList()，由adv_hypernet_layers参数控制层数
            key_extractors: 输入shape(...,state_dim), 输出shape(...,1)
            agents_extractors: 输入shape(...,state_dim), 输出shape(...,n_agents)
            action_extractors: 输入shape(...,state_action_dim), 输出shape(...,n_agents)

    '''
    def __init__(self, args):
        super(DMAQ_SI_Weight, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.state_dim = int(np.prod(args.state_shape))     # np.prod是所有元素之积
        self.action_dim = args.n_agents * self.n_actions
        self.state_action_dim = self.state_dim + self.action_dim

        self.num_kernel = args.num_kernel

        self.key_extractors = nn.ModuleList()
        self.agents_extractors = nn.ModuleList()
        self.action_extractors = nn.ModuleList()

        adv_hypernet_embed = self.args.adv_hypernet_embed   # embedding层的隐藏层数
        for i in range(self.num_kernel):  # multi-head attention
            if getattr(args, "adv_hypernet_layers", 1) == 1:
                self.key_extractors.append(nn.Linear(self.state_dim, 1))  # key
                self.agents_extractors.append(nn.Linear(self.state_dim, self.n_agents))  # agent
                self.action_extractors.append(nn.Linear(self.state_action_dim, self.n_agents))  # action
            elif getattr(args, "adv_hypernet_layers", 1) == 2:
                self.key_extractors.append(nn.Sequential(nn.Linear(self.state_dim, adv_hypernet_embed),
                                                         nn.ReLU(),
                                                         nn.Linear(adv_hypernet_embed, 1)))  # key
                self.agents_extractors.append(nn.Sequential(nn.Linear(self.state_dim, adv_hypernet_embed),
                                                            nn.ReLU(),
                                                            nn.Linear(adv_hypernet_embed, self.n_agents)))  # agent
                self.action_extractors.append(nn.Sequential(nn.Linear(self.state_action_dim, adv_hypernet_embed),
                                                            nn.ReLU(),
                                                            nn.Linear(adv_hypernet_embed, self.n_agents)))  # action
            elif getattr(args, "adv_hypernet_layers", 1) == 3:
                self.key_extractors.append(nn.Sequential(nn.Linear(self.state_dim, adv_hypernet_embed),
                                                         nn.ReLU(),
                                                         nn.Linear(adv_hypernet_embed, adv_hypernet_embed),
                                                         nn.ReLU(),
                                                         nn.Linear(adv_hypernet_embed, 1)))  # key
                self.agents_extractors.append(nn.Sequential(nn.Linear(self.state_dim, adv_hypernet_embed),
                                                            nn.ReLU(),
                                                            nn.Linear(adv_hypernet_embed, adv_hypernet_embed),
                                                            nn.ReLU(),
                                                            nn.Linear(adv_hypernet_embed, self.n_agents)))  # agent
                self.action_extractors.append(nn.Sequential(nn.Linear(self.state_action_dim, adv_hypernet_embed),
                                                            nn.ReLU(),
                                                            nn.Linear(adv_hypernet_embed, adv_hypernet_embed),
                                                            nn.ReLU(),
                                                            nn.Linear(adv_hypernet_embed, self.n_agents)))  # action
            else:
                raise Exception("Error setting number of adv hypernet layers.")

    def forward(self, states, actions):
        states = states.reshape(-1, self.state_dim)
        actions = actions.reshape(-1, self.action_dim)
        data = th.cat([states, actions], dim=1)

        # 经过embedding之后的输出(注意输出套了一层list)
        all_head_key = [k_ext(states) for k_ext in self.key_extractors]         # [shape(...,1)]
        all_head_agents = [k_ext(states) for k_ext in self.agents_extractors]   # [shape(...,n_agents)]
        all_head_action = [sel_ext(data) for sel_ext in self.action_extractors] # [shape(...,n_agents)]

        head_attend_weights = []
        for curr_head_key, curr_head_agents, curr_head_action in zip(all_head_key, all_head_agents, all_head_action):
            x_key = th.abs(curr_head_key).repeat(1, self.n_agents) + 1e-10 #v_{k}(\tau)
            x_agents = th.sigmoid(curr_head_agents) #\phi_{i, k}(\tau)
            x_action = th.sigmoid(curr_head_action) #\lambda_{i, k}(\tau, a)
            weights = x_key * x_agents * x_action #权重
            head_attend_weights.append(weights)

        head_attend = th.stack(head_attend_weights, dim=1)
        head_attend = head_attend.view(-1, self.num_kernel, self.n_agents)
        head_attend = th.sum(head_attend, dim=1)    #求和

        return head_attend

class DMAQer(nn.Module):
    '''
    Dueling Mixing Net(包含Transformation Net部分)
        Transformation Net: 混合全局信息state把Q_i(τ_i,a_i)和V_i(τ_i,a_i)转换为Q_i(τ,a_i)和V_i(τ,a_i)
        Dueling Mixing Net: 利用Q_i(τ,a_i)和V_i(τ,a_i)计算q_tot，v_tot (同时也要混合全局信息state)

    参数
        state_dim: 是多个state的维度之积
        action_dim: 包含了agent_id和n_actions的维度之和

    '''
    def __init__(self, args):
        super(DMAQer, self).__init__()
        # 智能体、环境参数读取
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.state_dim = int(np.prod(args.state_shape))
        self.action_dim = args.n_agents * self.n_actions
        self.state_action_dim = self.state_dim + self.action_dim + 1

        self.embed_dim = args.mixing_embed_dim  # 隐层维度

        hypernet_embed = self.args.hypernet_embed  # 超网络的隐层维度

        # Transformation Net的参数，weight与bias要混合全局信息state
        # 超网络输出权重 W(之后reshape成(1, n_agents))
        self.hyper_w_final = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                           nn.ReLU(),
                                           nn.Linear(hypernet_embed, self.n_agents))
        # 超网络输出偏差 bias
        self.V = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                               nn.ReLU(),
                               nn.Linear(hypernet_embed, self.n_agents))
        # 计算lambda
        self.si_weight = DMAQ_SI_Weight(args)

    # Q_{tot}=V_{tot}+A_{tot}=\sum Q_{i}+ \sum{(\lambda-1)*A_{i}}
    def calc_v(self, agent_qs):  # Dueling Mixing网络计算V_tot V_{tot}=\sum V_{i}
        '''
        计算v_tot = sum(q_i)，即每个agent的所选动作q值之和

        参数：
            agent_qs：是已经混合了state信息的Q_i(τ,a_i), 是每个agent所选择action的q值（标量）, (n_episodes, max_episode_len, n_agents)
        '''
        agent_qs = agent_qs.view(-1, self.n_agents)
        v_tot = th.sum(agent_qs, dim=-1)  # 求和
        return v_tot

    def calc_adv(self, agent_qs, states, actions, max_q_i):  # Dueling Mixing网络计算A_tot \sum{(\lambda-1)*A_{i}}
        '''
        计算A_tot

        参数：
            agent_qs：是已经混合了state信息的Q_i(τ,a_i), 是每个agent所选择action的q值（标量）, (n_episodes, max_episode_len, n_agents)
            states: (n_episodes, max_episode_len, state_shape) #TODO
            actions: (n_episodes, max_episode_len, n_agents, n_actions), 即当前agent_qs对应的action one hot
            max_q_i: 是已经混合了state信息的V_i(τ,a_i), 和agent_qs形状一样，只不过取的最优动作的q值, (n_episodes, max_episode_len, n_agents)
        '''
        states = states.reshape(-1, self.state_dim)
        actions = actions.reshape(-1, self.action_dim)
        agent_qs = agent_qs.view(-1, self.n_agents)
        max_q_i = max_q_i.view(-1, self.n_agents)

        adv_q = (agent_qs - max_q_i).view(-1, self.n_agents).detach()  # 计算优势函数, 即Q-V=A. 并去掉梯度

        adv_w_final = self.si_weight(states, actions)  # 获得权重lambda
        adv_w_final = adv_w_final.view(-1, self.n_agents)

        # 计算A_tot(两种形式：λ-1或者λ)
        if self.args.is_minus_one:  # 是不是相减的形式
            adv_tot = th.sum(adv_q * (adv_w_final - 1.), dim=1)  # \sum{(\lambda-1)*A_{i}}
        else:
            adv_tot = th.sum(adv_q * adv_w_final, dim=1)
        return adv_tot

    def calc(self, agent_qs, states, actions=None, max_q_i=None, is_v=False):  # 计算total价值函数
        if is_v:
            v_tot = self.calc_v(agent_qs)
            return v_tot
        else:
            adv_tot = self.calc_adv(agent_qs, states, actions, max_q_i)
            return adv_tot

    def forward(self, agent_qs, states, actions=None, max_q_i=None, is_v=False):
        '''
            参数：
                这里的agent_qs和max_q_i都是没有混合全局信息state，即Q_i(τ_i,a_i)和V_i(τ_i,a_i)，下面Transformation Net会混合
                is_v: 表示是否计算v_tot(否对应计算a_tot)，和个体v值无关

        '''
        bs = agent_qs.size(0)  # 样本数量数
        states = states.reshape(-1, self.state_dim)
        agent_qs = agent_qs.view(-1, self.n_agents)
        # 根据全局状态s获得transformation网络的参数
        w_final = self.hyper_w_final(states)  # 获得W权重
        w_final = th.abs(w_final)  # 求绝对值保证单调
        w_final = w_final.view(-1, self.n_agents) + 1e-10
        v = self.V(states)  # 获得b偏差
        v = v.view(-1, self.n_agents)

        # Transformation Net ( 获得Q_i(τ,a_i)或者V_i(τ,a_i), max_q_i是V_i(τ,a_i), agent_qs是Q_i(τ,a_i), 这里混合了全局信息所以是τ而不是τ_i )
        if self.args.weighted_head:  # 是否使用加权头
            agent_qs = w_final * agent_qs + v  # 计算智能体动作价值函数

        if not is_v:    # 计算individual v值V_i(τ,a_i)，计算A值时要用，所以这里not is_v表示不是计算v_tot
            max_q_i = max_q_i.view(-1, self.n_agents)
            if self.args.weighted_head:
                max_q_i = w_final * max_q_i + v  # 根据状态值函数计算

        # 若是v，则直接相加得到v_tot; 若不是，则得到A_tot
        y = self.calc(agent_qs, states, actions=actions, max_q_i=max_q_i, is_v=is_v)  # 进入Dueling Mixing网络，计算total

        v_tot = y.view(bs, -1, 1)

        return v_tot    # (bs, , 1)返回的是V(τ,a)或者A(τ,a)即tot


###############################################################
####################      QTran Mixer   #######################
###############################################################
# counterfactual joint networks, 输入state、所有agent的hidden_state、其他agent的动作、自己的编号，输出自己所有动作对应的联合Q值
class QtranQAlt(nn.Module):
    def __init__(self, args):
        super(QtranQAlt, self).__init__()
        self.args = args

        # 对每个agent的action进行编码
        self.action_encoding = nn.Sequential(nn.Linear(self.args.n_actions, self.args.n_actions),
                                             nn.ReLU(),
                                             nn.Linear(self.args.n_actions, self.args.n_actions))

        # 对每个agent的hidden_state进行编码
        self.hidden_encoding = nn.Sequential(nn.Linear(self.args.rnn_hidden_dim, self.args.rnn_hidden_dim),
                                             nn.ReLU(),
                                             nn.Linear(self.args.rnn_hidden_dim, self.args.rnn_hidden_dim))

        # 编码求和之后输入state、所有agent的hidden_state之和、其他agent的action之和, state包括当前agent的编号
        q_input = self.args.state_shape + self.args.n_actions + self.args.rnn_hidden_dim + self.args.n_agents
        self.q = nn.Sequential(nn.Linear(q_input, self.args.qtran_hidden_dim),
                               nn.ReLU(),
                               nn.Linear(self.args.qtran_hidden_dim, self.args.qtran_hidden_dim),
                               nn.ReLU(),
                               nn.Linear(self.args.qtran_hidden_dim, self.args.n_actions))

    # 因为所有时刻所有agent的hidden_states在之前已经计算好了，所以联合Q值可以一次计算所有transition的，不需要一条一条计算。
    def forward(self, state, hidden_states, actions):  # (episode_num, max_episode_len, n_agents, n_actions)
        # state的shape为(episode_num, max_episode_len, n_agents, state_shape+n_agents)，包括了当前agent的编号
        episode_num, max_episode_len, n_agents, n_actions = actions.shape

        # 对每个agent的action进行编码
        action_encoding = self.action_encoding(actions.reshape(-1, n_actions))
        action_encoding = action_encoding.reshape(episode_num, max_episode_len, n_agents, n_actions)

        # 对每个agent的hidden_state进行编码
        hidden_encoding = self.hidden_encoding(hidden_states.reshape(-1, self.args.rnn_hidden_dim))
        hidden_encoding = hidden_encoding.reshape(episode_num, max_episode_len, n_agents, self.args.rnn_hidden_dim)

        # 所有agent的hidden_encoding相加
        hidden_encoding = hidden_encoding.sum(dim=-2)  # (episode_num, max_episode_len, rnn_hidden_dim)
        hidden_encoding = hidden_encoding.unsqueeze(-2).expand(-1, -1, n_agents, -1)  # (episode_num, max_episode_len, n_agents， rnn_hidden_dim)

        # 对于每个agent，其他agent的action_encoding相加
        # 先让最后一维包含所有agent的动作
        action_encoding = action_encoding.reshape(episode_num, max_episode_len, 1, n_agents * n_actions)    # (episode_num, max_episode_len, n_agents， n_agents * n_actions)
        action_encoding = action_encoding.repeat(1, 1, n_agents, 1)  # 此时每个agent都有了所有agent的动作
        # 把每个agent自己的动作置0
        action_mask = (1 - th.eye(n_agents))  # th.eye（）生成一个二维对角矩阵
        action_mask = action_mask.view(-1, 1).repeat(1, n_actions).view(n_agents, -1)   # (n_agents， n_agents * n_actions)
        if self.args.cuda:
            action_mask = action_mask.cuda()
        action_encoding = action_encoding * action_mask.unsqueeze(0).unsqueeze(0)    # (1, 1, n_agents， n_agents * n_actions)之后会自动广播到episode_num, max_episode_len
        # 因为现在所有agent的动作都在最后一维，不能直接加。所以先扩展一维，相加后再去掉
        action_encoding = action_encoding.reshape(episode_num, max_episode_len, n_agents, n_agents, n_actions)
        action_encoding = action_encoding.sum(dim=-2)  # (episode_num, max_episode_len, n_agents， rnn_hidden_dim)

        inputs = th.cat([state, hidden_encoding, action_encoding], dim=-1)
        q_tot = self.q(inputs)
        return q_tot


# Joint action-value network， 输入state,所有agent的hidden_state，所有agent的动作，输出对应的联合Q值
class QtranQBase(nn.Module):
    '''
    得到joint q values

    '''
    def __init__(self, args):
        super(QtranQBase, self).__init__()
        self.args = args
        # action_encoding对输入的每个agent的hidden_state和动作进行编码，从而将所有agents的hidden_state和动作相加得到近似的联合hidden_state和动作
        ae_input = self.args.rnn_hidden_dim + self.args.n_actions
        self.hidden_action_encoding = nn.Sequential(nn.Linear(ae_input, ae_input),
                                             nn.ReLU(),
                                             nn.Linear(ae_input, ae_input))

        # 编码求和之后输入state、所有agent的hidden_state和动作之和
        q_input = self.args.state_shape + self.args.n_actions + self.args.rnn_hidden_dim
        self.q = nn.Sequential(nn.Linear(q_input, self.args.qtran_hidden_dim),
                               nn.ReLU(),
                               nn.Linear(self.args.qtran_hidden_dim, self.args.qtran_hidden_dim),
                               nn.ReLU(),
                               nn.Linear(self.args.qtran_hidden_dim, 1))

    # 因为所有时刻所有agent的hidden_states在之前已经计算好了，所以联合Q值可以一次计算所有transition的，不需要一条一条计算。
    def forward(self, state, hidden_states, actions):  # (episode_num, max_episode_len, n_agents, n_actions/hidden_dim)
        episode_num, max_episode_len, n_agents, _ = actions.shape
        hidden_actions = th.cat([hidden_states, actions], dim=-1)    # (episode_num, max_episode_len, n_agents, h_dim + n_actions)
        hidden_actions = hidden_actions.reshape(-1, self.args.rnn_hidden_dim + self.args.n_actions) # (episode_num * max_episode_len * n_agents, h_dim + n_actions)
        hidden_actions_encoding = self.hidden_action_encoding(hidden_actions)
        hidden_actions_encoding = hidden_actions_encoding.reshape(episode_num * max_episode_len, n_agents, -1)  # 变回n_agents维度用于求和
        hidden_actions_encoding = hidden_actions_encoding.sum(dim=-2)

        inputs = th.cat([state.reshape(episode_num * max_episode_len, -1), hidden_actions_encoding], dim=-1)
        q_tot = self.q(inputs)
        return q_tot    # (episode_num * max_episode_len, 1)


# 输入当前的state与所有agent的hidden_state, 输出V值
class QtranV(nn.Module):
    def __init__(self, args):
        super(QtranV, self).__init__()
        self.args = args

        # hidden_encoding对输入的每个agent的hidden_state编码，从而将所有agents的hidden_state相加得到近似的联合hidden_state
        hidden_input = self.args.rnn_hidden_dim
        self.hidden_encoding = nn.Sequential(nn.Linear(hidden_input, hidden_input),
                                             nn.ReLU(),
                                             nn.Linear(hidden_input, hidden_input))

        # 编码求和之后输入state、所有agent的hidden_state之和
        v_input = self.args.state_shape + self.args.rnn_hidden_dim
        self.v = nn.Sequential(nn.Linear(v_input, self.args.qtran_hidden_dim),
                               nn.ReLU(),
                               nn.Linear(self.args.qtran_hidden_dim, self.args.qtran_hidden_dim),
                               nn.ReLU(),
                               nn.Linear(self.args.qtran_hidden_dim, 1))

    def forward(self, state, hidden):
        episode_num, max_episode_len, n_agents, _ = hidden.shape    # (episode_num, max_episode_len, n_agents, rnn_hidden_dim)
        state = state.reshape(episode_num * max_episode_len, -1)    # (episode_num * max_episode_len, state_dim)
        hidden_encoding = self.hidden_encoding(hidden.reshape(-1, self.args.rnn_hidden_dim))
        hidden_encoding = hidden_encoding.reshape(episode_num * max_episode_len, n_agents, -1).sum(dim=-2)
        inputs = th.cat([state, hidden_encoding], dim=-1)
        v = self.v(inputs)
        return v
