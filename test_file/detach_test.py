import torch as th
import numpy as np
from network.q_network import RNNQNet

a = th.tensor([1,1,1], dtype=th.float32)
b = th.tensor([2,2,2], dtype=th.float32)

model = th.nn.Linear(3, 1)
model.weight.data.fill_(1)
model.bias.data.fill_(0)


test = 3

# 使用两次模型
c = model(a)
print(c)

d = model(b)
print(d)
if test == 0:
    c = c.detach()
    out = c + d         # out会detach一部分
elif test == 1:
    c_ = c.detach()     # 只有c_会detach，c没有detach，out用的c，梯度会全部保留
    out = c + d
elif test == 2:
    c_ = c.detach()     # 只有c_会detach，c没有detach。out用的c_，则out会detach一部分
    out = c_ + d
elif test == 3:
    c_ = c.detach_()    # 就地detach，c和c_都detach
    out = c + d



out.sum().backward()
print(model.weight.grad)

'''
结论：
    中间使用模型2次，其中只要1次的输出作为loss求梯度。只要loss运算使用的是那一次的输出没被detach的，其他模型输出在loss运算中都是detach的，那么就可以
'''