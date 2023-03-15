import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch as th

# 预测环境下一个时刻的信息
class WorldModel(nn.Module):
    def __init__(self, args):
        super(WorldModel, self).__init__()
        self.args = args

        # net
        self.hidden_embd = nn.Sequential(nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim),
                                         nn.ReLU(),
                                         nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim)
                                         )

        self.r_out = nn.Linear(args.rnn_hidden_dim, args.n_actions)
        self.o_out = nn.Linear(args.rnn_hidden_dim, args.obs_shape)
        self.terminate_out = nn.Linear(args.rnn_hidden_dim, 2)

    def forward(self, hidden_state):
        '''
        参数:
            hidden:自己的hidden
            u_agent:所有agent的action

        输出:
            reward:自己未选定动作时别人的action对环境的影响
            o_next:下个时刻自己的观测
        '''
        h_embedding = F.relu(self.hidden_embd(hidden_state))

        inps = h_embedding

        reward = self.r_out(inps)
        o_next = self.o_out(inps)
        terminated = self.terminate_out(inps)


        return reward, o_next, terminated


class Agent(nn.Module):
    def __init__(self, input_shape, args):
        super(Agent, self).__init__()
        self.args = args

        # RNN Q Net
        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)  # 只用的一个cell
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

        # World Model
        self.world = WorldModel(args)

    def forward(self, obs, hidden_state):
        # episode_num * n_agents作为bs
        x = F.relu(self.fc1(obs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)

        r, o_next, terminated = self.world(h)

        returns = {}
        returns["hidden_state"] = h
        returns["r"] = r
        returns["o_next"] = o_next
        returns["terminated"] = terminated
        q += r

        return q, returns  # q值是一个n_actions的向量


# 自己对其他人action预测
class TeammateModel(nn.Module):
    def __init__(self, args):
        super(TeammateModel, self).__init__()

        self.args = args
        self.hidden_dim = args.hidden_dim
        self.hidden_state_dim = args.hidden_state_dim
        self.obs_shape = args.obs_shape
        self.state_shape = args.state_shape
        self.n_actions = self.args.n_actions

        # net
        self.obs_embd = nn.Linear(self.obs_shape, self.hidden_dim)
        self.hidden_embd = nn.Linear(self.hidden_state_dim, self.hidden_dim)
        self.action_embd = nn.Linear(self.n_actions, self.hidden_dim)

        self.fc = nn.Linear(self.hidden_dim * 3, self.hidden_dim)

        self.r_out = nn.Linear(self.hidden_dim, 1)
        self.s_out = nn.Linear(self.hidden_dim, self.state_shape)
        self.terminate_out = nn.Linear(self.hidden_dim, 2)

    def forward(self, obs, hidden_state, actions):
        o_embedding = F.relu(self.obs_embd(obs))
        h_embedding = F.relu(self.hidden_embd(hidden_state))
        a_embedding = F.relu(self.action_embd(actions))

        inps = th.cat([o_embedding, h_embedding, a_embedding], dim=-1)

        reward = F.relu(self.r_out(inps))
        state_next = F.relu(self.s_out(inps))
        terminated = F.sigmoid(self.terminate_out(inps))

        return reward, state_next, terminated


# 利用Transformer产生消息
class MessageGenerator(nn.Module):
    def __init__(self):
        super(MessageGenerator, self).__init__()
        pass


