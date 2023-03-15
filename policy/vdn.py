import numpy as np
from network.q_network import RNNQNetwork
import torch as th


class VDNPolicy():
    '''  '''
    def __init__(self, args):
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        input_shape = self.obs_shape
        # 根据参数决定RNN的输入维度
        if args.last_action:  # 是否使用上一次的action作为输入
            input_shape += self.n_actions
        if args.reuse_network:  # 是否所有agent共用一个网络，若是则增加一个one hot输入标识身份
            input_shape += self.n_agents
        # 可能的最终输入：obs，上一次action，one hot的agent标识拼接得到


        self.q_net = RNNQNetwork(input_shape, args)


