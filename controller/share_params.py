import torch as th
import numpy as np
from network.q_network import RNNQNet, RNNQNetWithState
from network.RTW import RTWAgent
from network.world_model import Agent

# This multi-agent controller shares parameters between agents
class SharedMAC:
    '''
    n个agent共享一个神经网络产生动作(需要把agent_id作为输入)
    作用：
        控制所有agent如何选择动作
        保存与加载模型

    需要：
        n个agent(若共享参数就是一个agent)

    '''
    def __init__(self, args):
        # initial args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        self.args = args

        # initial agent input shape
        input_shape = self._get_input_shape()


        # build shared params agent
        self._build_agents(input_shape)

        # 要为每个agent维护一个hidden_state(mac可作为eval与target网络，两个网络各自都要用hidden_states)
        self.hidden_states = None   # (n_episodes, n_agents, hidden_dim)

    def choose_action(self, obs, last_action, agent_num, avail_actions, epsilon, evaluate=False):
        ''' 根据某个agent的obs等信息，为某个agent选择一个动作 '''
        # 计算Q值，并用epsilon-greedy选择action（必须选择可选动作）
        inputs = obs.copy()
        avail_actions_ind = np.nonzero(avail_actions)[0]  # index of actions which can be choose

        # transform agent_num to onehot vector
        agent_id = np.zeros(self.n_agents)
        agent_id[agent_num] = 1.

        if self.args.last_action:
            inputs = np.hstack((inputs, last_action))
        if self.args.reuse_network:
            inputs = np.hstack((inputs, agent_id))


        hidden_state = self.hidden_states[:, agent_num, :]

        # transform the shape of inputs from (42,) to (1,42)
        inputs = th.tensor(inputs, dtype=th.float32).unsqueeze(0)
        avail_actions = th.tensor(avail_actions, dtype=th.float32).unsqueeze(0)
        if self.args.cuda:
            inputs = inputs.cuda()
            hidden_state = hidden_state.cuda()

        # get q value 并更新hidden_state
        q_value, self.hidden_states[:, agent_num, :] = self.agent(inputs, hidden_state)

        # choose action from q value
        q_value[avail_actions == 0.0] = - float("inf")  # action不可用的赋值为-∞
        if np.random.uniform() < epsilon:
            action = np.random.choice(avail_actions_ind)  # action是一个整数
        else:
            action = th.argmax(q_value)

        return action

    def init_hidden(self, episode_num):
        # 为每个episode中的每个agent都初始化一个hidden states
        self.hidden_states = th.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))

    def cuda(self):
        self.agent.cuda()

    def _build_agents(self, input_shape):
        self.agent = RNNQNet(input_shape, self.args)

    def _build_inputs(self, batch, transition_idx):
        ''' 构建eval和target网络的输入 '''
        # 取出所有episode上该transition_idx的经验，u_onehot要取出所有，因为要用到上一条
        # 对于使用同一个网络的所有agent，还要把one hot agentID加在数据中
        obs, obs_next, u_onehot = batch['o'][:, transition_idx], \
                                  batch['o_next'][:, transition_idx], batch['u_onehot'][:]
        episode_num = obs.shape[0]
        inputs, inputs_next = [], []
        inputs.append(obs)
        inputs_next.append(obs_next)

        # 给obs添加上一个动作、agent编号
        if self.args.last_action:
            if transition_idx == 0:  # 如果是第一条经验，就让前一个动作为0向量
                inputs.append(th.zeros_like(u_onehot[:, transition_idx]))
            else:
                inputs.append(u_onehot[:, transition_idx - 1])
            inputs_next.append(u_onehot[:, transition_idx])
        if self.args.reuse_network:
            # 因为当前的obs三维的数据，每一维分别代表(episode，agent，obs维度)，直接在dim_1上添加对应的向量
            # 即可，比如给agent_0后面加(1, 0, 0, 0, 0)，表示5个agent中的0号。而agent_0的数据正好在第0行，那么需要加的
            # agent编号恰好就是一个单位矩阵，即对角线为1，其余为0
            inputs.append(th.eye(self.args.n_agents).unsqueeze(0).expand(episode_num, -1, -1))
            inputs_next.append(th.eye(self.args.n_agents).unsqueeze(0).expand(episode_num, -1, -1))
        # 要把obs中的三个拼起来，并且要把episode_num个episode、self.args.n_agents个agent的数据拼成episode_num*n_agents条数据
        # 因为这里所有agent共享一个神经网络，每条数据中带上了自己的编号，所以还是自己的数据
        inputs = th.cat([x.reshape(episode_num * self.args.n_agents, -1) for x in inputs], dim=1)
        inputs_next = th.cat([x.reshape(episode_num * self.args.n_agents, -1) for x in inputs_next], dim=1)
        return inputs, inputs_next  # 都为(episode_num * n_agents, obs + num_actions + n_agents)

    def _get_input_shape(self):
        '''  用于初始化RNN网络中的第一个Linear层的输入维度 '''
        input_shape = self.obs_shape
        # 根据参数决定RNN的输入维度
        if self.args.last_action:  # 是否使用上一次的action作为输入
            input_shape += self.n_actions
        if self.args.reuse_network:  # 是否所有agent共用一个网络，若是则增加一个one hot输入标识身份
            input_shape += self.n_agents
        # 可能的最终输入：obs，上一次action，one hot的agent标识拼接得到
        return input_shape

    def get_current_q_values(self, batch, max_episode_len):
        ''' 获取一个batch中所有transition的current action的q_values '''
        episode_num = batch['o'].shape[0]
        q_values = []
        ep_hidden_states = []
        for transition_idx in range(max_episode_len):
            inputs, _ = self._build_inputs(batch, transition_idx)  #(episode_num * n_agents, obs + num_actions + n_agents)   取得多个episode中transition_idx的数据(episode_num, 1, n_agents, features)，给obs加last_action、agent_id
            if self.args.cuda:
                inputs = inputs.cuda()
                self.hidden_states = self.hidden_states.cuda()  # 更新hidden_state(初始化为全0)
            q_value, self.hidden_states = self.agent(inputs, self.hidden_states)  # 得到的q_values维度为(episode_num*n_agents, n_actions)

            # 把q_values维度重新变回(episode_num, n_agents, n_actions)
            q_value = q_value.view(episode_num, self.n_agents, -1)
            q_values.append(q_value)

            ep_hidden_states.append(self.hidden_states.view(episode_num, self.n_agents, -1))
        # 得的q_values是一个列表，列表里装着max_episode_len个数组，数组的的维度是(episode个数, n_agents，n_actions)
        # 把该列表转化成(episode个数, max_episode_len， n_agents，n_actions)的数组
        q_values = th.stack(q_values, dim=1)
        ep_hidden_states = th.stack(ep_hidden_states, dim=1)
        return q_values, ep_hidden_states  # (episode个数, max_episode_len， n_agents，n_actions)

    def get_next_q_values(self, batch, max_episode_len):
        ''' 获取一个batch中所有transition的next action的q_values '''
        episode_num = batch['o_next'].shape[0]
        q_values = []
        ep_hidden_states = []
        for transition_idx in range(max_episode_len):
            _, inputs = self._build_inputs(batch, transition_idx)  # (episode_num * n_agents, obs + num_actions + n_agents)   取得多个episode中transition_idx的数据(episode_num, 1, n_agents, features)，给obs加last_action、agent_id
            if self.args.cuda:
                inputs = inputs.cuda()
                self.hidden_states = self.hidden_states.cuda()  # 更新hidden_state(初始化为全0)
            q_value, self.hidden_states = self.agent(inputs, self.hidden_states)  # 得到的q_values维度为(episode_num*n_agents, n_actions)

            ep_hidden_states.append(self.hidden_states.view(episode_num, self.n_agents, -1))
            # 把q_values维度重新变回(episode_num, n_agents, n_actions)
            q_value = q_value.view(episode_num, self.n_agents, -1)
            q_values.append(q_value)
        # 得的q_values是一个列表，列表里装着max_episode_len个数组，数组的的维度是(episode个数, n_agents，n_actions)
        # 把该列表转化成(episode个数, max_episode_len， n_agents，n_actions)的数组
        q_values = th.stack(q_values, dim=1)
        ep_hidden_states = th.stack(ep_hidden_states, dim=1)
        return q_values, ep_hidden_states  # (episode个数, max_episode_len， n_agents，n_actions)


    def parameters(self):
        return self.agent.parameters()

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def save_models(self, path):
        th.save(self.agent.state_dict(), path)

    def load_models(self, path):
        map_location = 'cuda:0' if self.args.cuda else 'cpu'
        self.agent.load_state_dict(th.load(path, map_location=map_location))

# This multi-agent controller shares parameters between agents
class SharedMACWithState:
    '''
    n个agent共享一个神经网络产生动作(需要把agent_id作为输入)
    作用：
        控制所有agent如何选择动作
        保存与加载模型

    需要：
        n个agent(若共享参数就是一个agent)

    '''
    def __init__(self, args):
        # initial args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        self.args = args

        # initial agent input shape
        input_shape = self._get_input_shape()


        # build shared params agent
        self._build_agents(input_shape)

        # 要为每个agent维护一个hidden_state(mac可作为eval与target网络，两个网络各自都要用hidden_states)
        self.hidden_states = None   # (n_episodes, n_agents, hidden_dim)

    def choose_action(self, obs, last_action, agent_num, avail_actions, epsilon, evaluate=False):
        ''' 根据某个agent的obs等信息，为某个agent选择一个动作 '''
        # 计算Q值，并用epsilon-greedy选择action（必须选择可选动作）
        inputs = obs.copy()
        avail_actions_ind = np.nonzero(avail_actions)[0]  # index of actions which can be choose

        # transform agent_num to onehot vector
        agent_id = np.zeros(self.n_agents)
        agent_id[agent_num] = 1.

        if self.args.last_action:
            inputs = np.hstack((inputs, last_action))
        if self.args.reuse_network:
            inputs = np.hstack((inputs, agent_id))


        hidden_state = self.hidden_states[:, agent_num, :]

        # transform the shape of inputs from (42,) to (1,42)
        inputs = th.tensor(inputs, dtype=th.float32).unsqueeze(0)
        avail_actions = th.tensor(avail_actions, dtype=th.float32).unsqueeze(0)
        if self.args.cuda:
            inputs = inputs.cuda()
            hidden_state = hidden_state.cuda()

        # get q value 并更新hidden_state
        q_value, returns = self.agent(inputs, hidden_state)
        self.hidden_states[:, agent_num, :] = returns["hidden_state"]

        # choose action from q value
        q_value[avail_actions == 0.0] = - float("inf")  # action不可用的赋值为-∞
        if np.random.uniform() < epsilon:
            action = np.random.choice(avail_actions_ind)  # action是一个整数
        else:
            action = th.argmax(q_value)

        return action

    def init_hidden(self, episode_num):
        # 为每个episode中的每个agent都初始化一个hidden states
        self.hidden_states = th.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))

    def cuda(self):
        self.agent.cuda()

    def _build_agents(self, input_shape):
        self.agent = Agent(input_shape, self.args)

    def _build_inputs(self, batch, transition_idx):
        ''' 构建eval和target网络的输入 '''
        # 取出所有episode上该transition_idx的经验，u_onehot要取出所有，因为要用到上一条
        # 对于使用同一个网络的所有agent，还要把one hot agentID加在数据中
        obs, obs_next, u_onehot = batch['o'][:, transition_idx], \
                                  batch['o_next'][:, transition_idx], batch['u_onehot'][:]
        episode_num = obs.shape[0]
        inputs, inputs_next = [], []
        inputs.append(obs)
        inputs_next.append(obs_next)

        # 给obs添加上一个动作、agent编号
        if self.args.last_action:
            if transition_idx == 0:  # 如果是第一条经验，就让前一个动作为0向量
                inputs.append(th.zeros_like(u_onehot[:, transition_idx]))
            else:
                inputs.append(u_onehot[:, transition_idx - 1])
            inputs_next.append(u_onehot[:, transition_idx])
        if self.args.reuse_network:
            # 因为当前的obs三维的数据，每一维分别代表(episode，agent，obs维度)，直接在dim_1上添加对应的向量
            # 即可，比如给agent_0后面加(1, 0, 0, 0, 0)，表示5个agent中的0号。而agent_0的数据正好在第0行，那么需要加的
            # agent编号恰好就是一个单位矩阵，即对角线为1，其余为0
            inputs.append(th.eye(self.args.n_agents).unsqueeze(0).expand(episode_num, -1, -1))
            inputs_next.append(th.eye(self.args.n_agents).unsqueeze(0).expand(episode_num, -1, -1))
        # 要把obs中的三个拼起来，并且要把episode_num个episode、self.args.n_agents个agent的数据拼成episode_num*n_agents条数据
        # 因为这里所有agent共享一个神经网络，每条数据中带上了自己的编号，所以还是自己的数据
        inputs = th.cat([x.reshape(episode_num * self.args.n_agents, -1) for x in inputs], dim=1)
        inputs_next = th.cat([x.reshape(episode_num * self.args.n_agents, -1) for x in inputs_next], dim=1)
        return inputs, inputs_next  # 都为(episode_num * n_agents, obs + num_actions + n_agents)

    def _get_input_shape(self):
        '''  用于初始化RNN网络中的第一个Linear层的输入维度 '''
        input_shape = self.obs_shape
        # 根据参数决定RNN的输入维度
        if self.args.last_action:  # 是否使用上一次的action作为输入
            input_shape += self.n_actions
        if self.args.reuse_network:  # 是否所有agent共用一个网络，若是则增加一个one hot输入标识身份
            input_shape += self.n_agents
        # 可能的最终输入：obs，上一次action，one hot的agent标识拼接得到
        return input_shape

    def get_current_q_values(self, batch, max_episode_len):
        ''' 获取一个batch中所有transition的current action的q_values '''
        episode_num = batch['o'].shape[0]
        q_values = []
        ep_hidden_states = []
        r = []
        o_next = []
        terminated = []
        for transition_idx in range(max_episode_len):
            inputs, _ = self._build_inputs(batch,
                                           transition_idx)  # (episode_num * n_agents, obs + num_actions + n_agents)   取得多个episode中transition_idx的数据(episode_num, 1, n_agents, features)，给obs加last_action、agent_id
            if self.args.cuda:
                inputs = inputs.cuda()
                self.hidden_states = self.hidden_states.cuda()  # 更新hidden_state(初始化为全0)
            q_value, returns = self.agent(inputs, self.hidden_states)  # 得到的q_values维度为(episode_num*n_agents, n_actions)
            self.hidden_states = returns["hidden_state"]

            # 把q_values维度重新变回(episode_num, n_agents, n_actions)
            q_value = q_value.view(episode_num, self.n_agents, -1)
            r.append(returns["r"].view(episode_num, self.n_agents, -1))
            o_next.append(returns["o_next"].view(episode_num, self.n_agents, -1))
            terminated.append(returns["terminated"].view(episode_num, self.n_agents, -1))
            q_values.append(q_value)
            ep_hidden_states.append(self.hidden_states.view(episode_num, self.n_agents, -1))
        # 得的q_values是一个列表，列表里装着max_episode_len个数组，数组的的维度是(episode个数, n_agents，n_actions)
        # 把该列表转化成(episode个数, max_episode_len， n_agents，n_actions)的数组
        q_values = th.stack(q_values, dim=1)
        ep_hidden_states = th.stack(ep_hidden_states, dim=1)
        returns = {}
        returns["ep_hidden_states"] = ep_hidden_states
        returns["r"] = th.stack(r, dim=1)
        returns["o_next"] = th.stack(o_next, dim=1)
        returns["terminated"] = th.stack(terminated, dim=1)

        return q_values, returns  # (episode个数, max_episode_len， n_agents，n_actions)

    def get_next_q_values(self, batch, max_episode_len):
        ''' 获取一个batch中所有transition的current action的q_values '''
        episode_num = batch['o_next'].shape[0]
        q_values = []
        ep_hidden_states = []
        r = []
        o_next = []
        terminated = []
        for transition_idx in range(max_episode_len):
            _, inputs = self._build_inputs(batch,
                                           transition_idx)  # (episode_num * n_agents, obs + num_actions + n_agents)   取得多个episode中transition_idx的数据(episode_num, 1, n_agents, features)，给obs加last_action、agent_id
            if self.args.cuda:
                inputs = inputs.cuda()
                self.hidden_states = self.hidden_states.cuda()  # 更新hidden_state(初始化为全0)
            q_value, returns = self.agent(inputs, self.hidden_states)  # 得到的q_values维度为(episode_num*n_agents, n_actions)
            self.hidden_states = returns["hidden_state"]

            # 把q_values维度重新变回(episode_num, n_agents, n_actions)
            q_value = q_value.view(episode_num, self.n_agents, -1)
            r.append(returns["r"].view(episode_num, self.n_agents, -1))
            o_next.append(returns["o_next"].view(episode_num, self.n_agents, -1))
            terminated.append(returns["terminated"].view(episode_num, self.n_agents, -1))
            q_values.append(q_value)
            ep_hidden_states.append(self.hidden_states.view(episode_num, self.n_agents, -1))
        # 得的q_values是一个列表，列表里装着max_episode_len个数组，数组的的维度是(episode个数, n_agents，n_actions)
        # 把该列表转化成(episode个数, max_episode_len， n_agents，n_actions)的数组
        q_values = th.stack(q_values, dim=1)
        ep_hidden_states = th.stack(ep_hidden_states, dim=1)
        returns = {}
        returns["ep_hidden_states"] = ep_hidden_states
        returns["r"] = th.stack(r, dim=1)
        returns["o_next"] = th.stack(o_next, dim=1)
        returns["terminated"] = th.stack(terminated, dim=1)

        return q_values, returns  # (episode个数, max_episode_len， n_agents，n_actions)


    def parameters(self):
        return self.agent.parameters()

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def save_models(self, path):
        th.save(self.agent.state_dict(), path)

    def load_models(self, path):
        map_location = 'cuda:0' if self.args.cuda else 'cpu'
        self.agent.load_state_dict(th.load(path, map_location=map_location))

class SeparatedMAC:
    '''
    每个agent都有自己的网络
    作用：
        控制所有agent如何选择动作
        保存与加载模型

    需要：
        n个agent

    '''
    def __init__(self, args):
        # initial args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        self.args = args

        # initial agent input shape
        input_shape = self._get_input_shape()


        # build separated params agent
        self._build_agents(input_shape)

        # 要为每个agent维护一个hidden_state(mac可作为eval与target网络，两个网络各自都要用hidden_states)
        self.hidden_states = None   # (n_episodes, n_agents, hidden_dim)

    def choose_action(self, obs, last_action, agent_num, avail_actions, epsilon, evaluate=False):
        ''' 根据某个agent的obs等信息，为某个agent选择一个动作 '''
        # 计算Q值，并用epsilon-greedy选择action（必须选择可选动作）
        inputs = obs.copy()
        avail_actions_ind = np.nonzero(avail_actions)[0]  # index of actions which can be choose

        # transform agent_num to onehot vector
        agent_id = np.zeros(self.n_agents)
        agent_id[agent_num] = 1.

        if self.args.last_action:
            inputs = np.hstack((inputs, last_action))
        if self.args.reuse_network:
            inputs = np.hstack((inputs, agent_id))


        hidden_state = self.hidden_states[:, agent_num, :]

        # transform the shape of inputs from (42,) to (1,42)
        inputs = th.tensor(inputs, dtype=th.float32).unsqueeze(0)
        avail_actions = th.tensor(avail_actions, dtype=th.float32).unsqueeze(0)
        if self.args.cuda:
            inputs = inputs.cuda()
            hidden_state = hidden_state.cuda()

        # get q value 并更新hidden_state
        q_value, self.hidden_states[:, agent_num, :] = self.agent[agent_num](inputs, hidden_state)

        # choose action from q value
        q_value[avail_actions == 0.0] = - float("inf")  # action不可用的赋值为-∞
        if np.random.uniform() < epsilon:
            action = np.random.choice(avail_actions_ind)  # action是一个整数
        else:
            action = th.argmax(q_value)

        return action

    def init_hidden(self, episode_num):
        # 为每个episode中的每个agent都初始化一个hidden states
        self.hidden_states = th.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))

    def cuda(self):
        for i in range(self.n_agents):
            self.agent[i].cuda()

    def _build_agents(self, input_shape):
        ''' n个agent都有自己的网络 '''
        self.agent = [RNNQNet(input_shape, self.args) for i in range(self.n_agents)]    #lst

    def _build_inputs(self, batch, transition_idx):
        ''' 构建eval和target网络的输入 '''
        # 取出所有episode上该transition_idx的经验，u_onehot要取出所有，因为要用到上一条
        # 对于使用同一个网络的所有agent，还要把one hot agentID加在数据中
        obs, obs_next, u_onehot = batch['o'][:, transition_idx], \
                                  batch['o_next'][:, transition_idx], batch['u_onehot'][:]
        episode_num = obs.shape[0]
        inputs, inputs_next = [], []
        inputs.append(obs)
        inputs_next.append(obs_next)

        # 给obs添加上一个动作、agent编号
        if self.args.last_action:
            if transition_idx == 0:  # 如果是第一条经验，就让前一个动作为0向量
                inputs.append(th.zeros_like(u_onehot[:, transition_idx]))
            else:
                inputs.append(u_onehot[:, transition_idx - 1])
            inputs_next.append(u_onehot[:, transition_idx])
        if self.args.reuse_network:
            # 因为当前的obs三维的数据，每一维分别代表(episode，agent，obs维度)，直接在dim_1上添加对应的向量
            # 即可，比如给agent_0后面加(1, 0, 0, 0, 0)，表示5个agent中的0号。而agent_0的数据正好在第0行，那么需要加的
            # agent编号恰好就是一个单位矩阵，即对角线为1，其余为0
            inputs.append(th.eye(self.args.n_agents).unsqueeze(0).expand(episode_num, -1, -1))
            inputs_next.append(th.eye(self.args.n_agents).unsqueeze(0).expand(episode_num, -1, -1))
        # 要把obs中的三个拼起来，并且要把episode_num个episode、self.args.n_agents个agent的数据拼成episode_num*n_agents条数据
        # 因为这里所有agent共享一个神经网络，每条数据中带上了自己的编号，所以还是自己的数据
        inputs = th.cat([x.reshape(episode_num * self.args.n_agents, -1) for x in inputs], dim=1)
        inputs_next = th.cat([x.reshape(episode_num * self.args.n_agents, -1) for x in inputs_next], dim=1)
        return inputs, inputs_next  # 都为(episode_num * n_agents, obs + num_actions + n_agents)

    def _get_input_shape(self):
        '''  用于初始化RNN网络中的第一个Linear层的输入维度 '''
        input_shape = self.obs_shape
        # 根据参数决定RNN的输入维度
        if self.args.last_action:  # 是否使用上一次的action作为输入
            input_shape += self.n_actions
        if self.args.reuse_network:  # 是否所有agent共用一个网络，若是则增加一个one hot输入标识身份
            input_shape += self.n_agents
        # 可能的最终输入：obs，上一次action，one hot的agent标识拼接得到
        return input_shape

    def get_current_q_values(self, batch, max_episode_len):
        ''' 获取一个batch中所有transition的current action的q_values '''
        episode_num = batch['o'].shape[0]
        q_values = []
        ep_hidden_states = []

        # 将每个agent的hidden_states分离，不然会报错
        hidden_states = []
        if self.args.cuda:
            self.hidden_states = self.hidden_states.cuda()
        for agent_id in range(self.n_agents):
            hidden_states.append(self.hidden_states[:, agent_id, :])

        for transition_idx in range(max_episode_len):
            inputs, _ = self._build_inputs(batch, transition_idx)  # (episode_num * n_agents, obs + num_actions)   取得多个episode中transition_idx的数据(episode_num, 1, n_agents, features)，给obs加last_action、agent_id

            # 每个agent不能作为batch，需要变换形状(之后变回来)
            inputs = inputs.view(episode_num, self.n_agents, -1)  # (episode_num, n_agents, obs + num_actions)

            if self.args.cuda:
                inputs = inputs.cuda()
                # self.hidden_states = self.hidden_states.cuda()  # 更新hidden_state(初始化为全0)
            q_value = []
            for agent_id in range(self.n_agents):
                q_value_agent, hidden_states[agent_id] = self.agent[agent_id](inputs[:, agent_id, :], hidden_states[agent_id])  # 得到的q_value_agent维度为(episode_num, n_actions)
                q_value.append(q_value_agent)
            ep_hidden_states.append(th.stack(hidden_states, dim=1))
            q_value = th.stack(q_value, dim=0)

            # 把q_values维度重新变回(episode_num, n_agents, n_actions)
            q_value = q_value.permute(1, 0, 2)
            q_values.append(q_value)

        # 更新hidden state
        for agent_id in range(self.n_agents):
            self.hidden_states[:, agent_id, :] = hidden_states[agent_id]

        # 得的q_values是一个列表，列表里装着max_episode_len个数组，数组的的维度是(episode个数, n_agents，n_actions)
        # 把该列表转化成(episode个数, max_episode_len， n_agents，n_actions)的数组
        q_values = th.stack(q_values, dim=1)
        ep_hidden_states = th.stack(ep_hidden_states, dim=1)
        return q_values, ep_hidden_states  # (episode个数, max_episode_len， n_agents，n_actions)

    def get_next_q_values(self, batch, max_episode_len):
        ''' 获取一个batch中所有transition的current action的q_values '''
        episode_num = batch['o_next'].shape[0]
        q_values = []
        ep_hidden_states =[]
        # 将每个agent的hidden_states分离，不然会报错
        hidden_states = []
        if self.args.cuda:
            self.hidden_states = self.hidden_states.cuda()
        for agent_id in range(self.n_agents):
            hidden_states.append(self.hidden_states[:, agent_id, :])

        for transition_idx in range(max_episode_len):
            _, inputs = self._build_inputs(batch, transition_idx)  # (episode_num * n_agents, obs + num_actions)   取得多个episode中transition_idx的数据(episode_num, 1, n_agents, features)，给obs加last_action、agent_id

            # 每个agent不能作为batch，需要变换形状(之后变回来)
            inputs = inputs.view(episode_num, self.n_agents, -1)  # (episode_num, n_agents, obs + num_actions)

            if self.args.cuda:
                inputs = inputs.cuda()
                # self.hidden_states = self.hidden_states.cuda()  # 更新hidden_state(初始化为全0)
            q_value = []
            for agent_id in range(self.n_agents):
                q_value_agent, hidden_states[agent_id] = self.agent[agent_id](inputs[:, agent_id, :], hidden_states[agent_id])  # 得到的q_value_agent维度为(episode_num, n_actions)
                q_value.append(q_value_agent)
            q_value = th.stack(q_value, dim=0)
            ep_hidden_states.append(th.stack(hidden_states, dim=1))

            # 把q_values维度重新变回(episode_num, n_agents, n_actions)
            q_value = q_value.permute(1, 0, 2)
            q_values.append(q_value)
        # 更新hidden state
        for agent_id in range(self.n_agents):
            self.hidden_states[:, agent_id, :] = hidden_states[agent_id]

        # 得的q_values是一个列表，列表里装着max_episode_len个数组，数组的的维度是(episode个数, n_agents，n_actions)
        # 把该列表转化成(episode个数, max_episode_len， n_agents，n_actions)的数组
        q_values = th.stack(q_values, dim=1)
        ep_hidden_states = th.stack(ep_hidden_states, dim=1)
        return q_values, hidden_states  # (episode个数, max_episode_len， n_agents，n_actions)


    def parameters(self):
        params = []
        for agent_id in range(self.n_agents):
            params += list(self.agent[agent_id].parameters())
        return params

    def load_state(self, other_mac):
        for agent_id in range(self.n_agents):
            self.agent[agent_id].load_state_dict(other_mac.agent.state_dict())

    def save_models(self, path):
        for agent_id in range(self.n_agents):
            th.save(self.agent[agent_id].state_dict(), path)

    def load_models(self, path):
        map_location = 'cuda:0' if self.args.cuda else 'cpu'
        for agent_id in range(self.n_agents):
            self.agent[agent_id].load_state_dict(th.load(path, map_location=map_location))

class RTWMAC:
    '''
    n个agent共享一个神经网络产生动作(需要把agent_id作为输入)
    作用：
        控制所有agent如何选择动作
        保存与加载模型

    需要：
        n个agent(若共享参数就是一个agent)

    '''
    def __init__(self, args):
        # initial args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        self.args = args

        # initial agent input shape
        input_shape = self._get_input_shape()


        # build shared params agent
        self._build_agents(input_shape)

        # 要为每个agent维护一个hidden_state(mac可作为eval与target网络，两个网络各自都要用hidden_states)
        self.hidden_states = None   # (n_episodes, n_agents, hidden_dim)

    def choose_action(self, obs, last_action, agent_num, avail_actions, epsilon, evaluate=False):
        ''' 根据某个agent的obs等信息，为某个agent选择一个动作 '''
        # 计算Q值，并用epsilon-greedy选择action（必须选择可选动作）
        inputs = obs.copy()
        avail_actions_ind = np.nonzero(avail_actions[agent_num])[0]  # index of actions which can be choose

        # transform agent_num to onehot vector
        agent_id = np.zeros(self.n_agents)
        agent_id[agent_num] = 1.

        if self.args.last_action:
            inputs = np.hstack((inputs, last_action))
        if self.args.reuse_network:
            inputs = np.hstack((inputs, agent_id))

        hidden_state = self.hidden_states[:, agent_num, :]

        # transform the shape of inputs from (42,) to (1,42)
        inputs = th.tensor(inputs, dtype=th.float32).unsqueeze(0)
        avail_actions = th.tensor(avail_actions, dtype=th.float32).unsqueeze(0)
        o = th.tensor(obs).unsqueeze(0)
        if self.args.cuda:
            inputs = inputs.cuda()
            hidden_state = hidden_state.cuda()
            o = o.cuda()

        # get q value 并更新hidden_state
        q_value, self.hidden_states[:, agent_num, :] = self.agent(inputs, hidden_state, o, None, None, avail_actions, test_mode=True, agent_num=agent_num)

        # choose action from q value
        q_value[avail_actions[:, agent_num] == 0.0] = - float("inf")  # action不可用的赋值为-∞
        if np.random.uniform() < epsilon:
            action = np.random.choice(avail_actions_ind)  # action是一个整数
        else:
            action = th.argmax(q_value)

        return action

    def init_hidden(self, episode_num):
        # 为每个episode中的每个agent都初始化一个hidden states
        self.hidden_states = th.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))

    def cuda(self):
        self.agent.cuda()

    def _build_agents(self, input_shape):
        self.agent = RTWAgent(input_shape, self.args)

    def _build_inputs(self, batch, transition_idx):
        ''' 构建eval和target网络的输入 '''
        # 取出所有episode上该transition_idx的经验，u_onehot要取出所有，因为要用到上一条
        # 对于使用同一个网络的所有agent，还要把one hot agentID加在数据中
        obs, obs_next, u_onehot = batch['o'][:, transition_idx], \
                                  batch['o_next'][:, transition_idx], batch['u_onehot'][:]
        episode_num = obs.shape[0]
        inputs, inputs_next = [], []
        inputs.append(obs)
        inputs_next.append(obs_next)

        # 给obs添加上一个动作、agent编号
        if self.args.last_action:
            if transition_idx == 0:  # 如果是第一条经验，就让前一个动作为0向量
                inputs.append(th.zeros_like(u_onehot[:, transition_idx]))
            else:
                inputs.append(u_onehot[:, transition_idx - 1])
            inputs_next.append(u_onehot[:, transition_idx])
        if self.args.reuse_network:
            # 因为当前的obs三维的数据，每一维分别代表(episode，agent，obs维度)，直接在dim_1上添加对应的向量
            # 即可，比如给agent_0后面加(1, 0, 0, 0, 0)，表示5个agent中的0号。而agent_0的数据正好在第0行，那么需要加的
            # agent编号恰好就是一个单位矩阵，即对角线为1，其余为0
            inputs.append(th.eye(self.args.n_agents).unsqueeze(0).expand(episode_num, -1, -1))
            inputs_next.append(th.eye(self.args.n_agents).unsqueeze(0).expand(episode_num, -1, -1))
        # 要把obs中的三个拼起来，并且要把episode_num个episode、self.args.n_agents个agent的数据拼成episode_num*n_agents条数据
        # 因为这里所有agent共享一个神经网络，每条数据中带上了自己的编号，所以还是自己的数据
        inputs = th.cat([x.reshape(episode_num * self.args.n_agents, -1) for x in inputs], dim=1)
        inputs_next = th.cat([x.reshape(episode_num * self.args.n_agents, -1) for x in inputs_next], dim=1)
        return inputs, inputs_next  # 都为(episode_num * n_agents, obs + num_actions + n_agents)

    def _get_input_shape(self):
        '''  用于初始化RNN网络中的第一个Linear层的输入维度 '''
        input_shape = self.obs_shape
        # 根据参数决定RNN的输入维度
        if self.args.last_action:  # 是否使用上一次的action作为输入
            input_shape += self.n_actions
        if self.args.reuse_network:  # 是否所有agent共用一个网络，若是则增加一个one hot输入标识身份
            input_shape += self.n_agents
        # 可能的最终输入：obs，上一次action，one hot的agent标识拼接得到
        return input_shape
    # TODO 修改适合于RTW的agent
    def get_current_q_values(self, batch, max_episode_len):
        ''' 获取一个batch中所有transition的current action的q_values '''
        episode_num = batch['o'].shape[0]
        q_values = []
        ep_hidden_states = []
        loss1_tot = 0
        loss2_tot = 0
        for transition_idx in range(max_episode_len):
            inputs, _ = self._build_inputs(batch, transition_idx)  #(episode_num * n_agents, obs + num_actions + n_agents)   取得多个episode中transition_idx的数据(episode_num, 1, n_agents, features)，给obs加last_action、agent_id
            obs = batch['o'][:, transition_idx].reshape(-1, self.args.obs_shape)
            obs_next = batch['o_next'][:, transition_idx].reshape(-1, self.args.obs_shape)
            action = batch['u'][:, transition_idx].reshape(-1, 1)
            avail_action = batch['avail_u'][:, transition_idx].reshape(-1, self.args.n_actions)

            if self.args.cuda:
                inputs = inputs.cuda()
                self.hidden_states = self.hidden_states.cuda()  # 更新hidden_state(初始化为全0)
                obs = obs.cuda()
                obs_next = obs_next.cuda()
                action = action.cuda()
                avail_action = avail_action.cuda()
            q_value, self.hidden_states, loss1, loss2 = self.agent(inputs, self.hidden_states, obs, obs_next, action, avail_action)  # 得到的q_values维度为(episode_num*n_agents, n_actions)

            loss1_tot += loss1
            loss2_tot += loss2
            # 把q_values维度重新变回(episode_num, n_agents, n_actions)
            q_value = q_value.view(episode_num, self.n_agents, -1)
            q_values.append(q_value)

            ep_hidden_states.append(self.hidden_states.view(episode_num, self.n_agents, -1))
        # 得的q_values是一个列表，列表里装着max_episode_len个数组，数组的的维度是(episode个数, n_agents，n_actions)
        # 把该列表转化成(episode个数, max_episode_len， n_agents，n_actions)的数组
        q_values = th.stack(q_values, dim=1)
        ep_hidden_states = th.stack(ep_hidden_states, dim=1)
        return q_values, ep_hidden_states, loss1_tot/max_episode_len, loss2_tot/max_episode_len  # (episode个数, max_episode_len， n_agents，n_actions)

    def get_next_q_values(self, batch, max_episode_len):
        ''' 获取一个batch中所有transition的current action的q_values '''
        episode_num = batch['o_next'].shape[0]
        q_values = []
        ep_hidden_states = []
        for transition_idx in range(max_episode_len):
            _, inputs = self._build_inputs(batch, transition_idx)  # (episode_num * n_agents, obs + num_actions + n_agents)   取得多个episode中transition_idx的数据(episode_num, 1, n_agents, features)，给obs加last_action、agent_id
            obs = batch['o_next'][:, transition_idx].reshape(-1, self.args.obs_shape)
            avail_action = batch['avail_u'][:, transition_idx].reshape(-1, self.args.n_actions)
            if self.args.cuda:
                inputs = inputs.cuda()
                self.hidden_states = self.hidden_states.cuda()  # 更新hidden_state(初始化为全0)
                obs = obs.cuda()
                avail_action = batch['avail_u'][:, transition_idx].reshape(-1, self.args.n_actions)
            q_value, self.hidden_states, _, _ = self.agent(inputs, self.hidden_states, obs, None, None, avail_action, target=True)  # 得到的q_values维度为(episode_num*n_agents, n_actions)

            # 把q_values维度重新变回(episode_num, n_agents, n_actions)
            q_value = q_value.view(episode_num, self.n_agents, -1)
            q_values.append(q_value)

            ep_hidden_states.append(self.hidden_states.view(episode_num, self.n_agents, -1))
        # 得的q_values是一个列表，列表里装着max_episode_len个数组，数组的的维度是(episode个数, n_agents，n_actions)
        # 把该列表转化成(episode个数, max_episode_len， n_agents，n_actions)的数组
        q_values = th.stack(q_values, dim=1)
        ep_hidden_states = th.stack(ep_hidden_states, dim=1)
        return q_values, ep_hidden_states  # (episode个数, max_episode_len， n_agents，n_actions)


    def parameters(self):
        return self.agent.parameters()

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def save_models(self, path):
        th.save(self.agent.state_dict(), path)

    def load_models(self, path):
        map_location = 'cuda:0' if self.args.cuda else 'cpu'
        self.agent.load_state_dict(th.load(path, map_location=map_location))