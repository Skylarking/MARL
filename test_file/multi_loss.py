import torch as th
import torch.nn as nn
x = th.tensor([1,2,3], dtype=th.float32)
layer1 = nn.Linear(3, 2)
layer2 = nn.Linear(2, 1)

layer1.weight.data[:] = 1
layer1.bias.data[:] = 0
layer2.weight.data[:] = 1
layer2.bias.data[:] = 0

print(*[(name, param) for name,param in layer1.named_parameters()])
print(*[(name, param) for name,param in layer2.named_parameters()])

y1 = th.tensor(1)
y2 = th.tensor(1)

y1_hat = layer1(x)
y2_hat = layer2(y1_hat.detach())    # TODO detach or not
print(y1_hat)
print(y2_hat)
loss1 = (y1_hat.mean()-y1) ** 2
loss2 = (y2_hat-y2) ** 2

loss = loss1 + loss2

print(layer1.weight.grad)
print(layer2.weight.grad)
loss.backward()

print(layer1.weight.grad)
print(layer2.weight.grad)

'''
    结论：loss相加不影响梯度，loss1和loss2使用不同的网络，那么其相加后并backward后，loss的梯度由于对loss1梯度为1，那么loss1只能对模型1的参数有梯度
'''

