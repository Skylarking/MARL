'''
rnn_test
初始化多个rnn网络，用一个hidden保存所有rnn的hidden，但是backward会报错
'''


import torch.nn.functional as F
import torch.nn as nn
import torch as th


class RNNQNet(nn.Module):
    def __init__(self, input_shape, rnn_hidden_dim):
        super(RNNQNet, self).__init__()
        self.rnn_hidden_dim = rnn_hidden_dim
        self.fc1 = nn.Linear(input_shape, rnn_hidden_dim)
        self.rnn = nn.GRUCell(rnn_hidden_dim, rnn_hidden_dim)
        self.fc2 = nn.Linear(rnn_hidden_dim, 4)

    def forward(self, obs, hidden_state):
        x = F.relu(self.fc1(obs))
        h_in = hidden_state.reshape(-1, self.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return q, h



class ma():
    def __init__(self, test_num=0):
        self.agent = [RNNQNet(2, 3) for i in range(2)]
        if test_num == 0:   # 两种初始化hidden的方式，第一种会报错
            self.hidden = th.zeros((2,2,3), dtype=th.float32)   # 这段代码会报错
        else:
            # 修改成这样就不会报错，hidden不能在一个上面，不知道为什么
            self.hidden = [th.zeros((2,3), dtype=th.float32) for i in range(2)]
        self.params = []
        for i in range(2):
            self.params += list(self.agent[i].parameters())
        self.optim = th.optim.Adam(self.params, lr=0.1)
    def train(self):
        inp1 = th.randint(0, 20, size=(2,2), dtype=th.float32)
        inp2 = th.randint(0, 20, size=(2,2), dtype=th.float32)
        inp = th.stack((inp1,inp2), dim=0)
        outs = []
        for i in range(2):
            out, self.hidden[i] = self.agent[i](inp[i], self.hidden[i])
            outs.append(out)
        outs = th.stack(outs, dim=0)
        self.optim.zero_grad()
        loss = outs.mean()
        loss.backward()
        self.optim.step()
        print("ok")
if __name__ == '__main__':
    ma = ma(test_num=0)
    ma.train()