import torch as th
import torch.nn as nn
x = th.tensor([1,2,3], dtype=th.float32)
layer1 = nn.Linear(3, 2)
layer2 = nn.Linear(2, 1)

opt1 = th.optim.SGD(params=layer1.parameters(), lr=1)
opt2 = th.optim.SGD(params=layer2.parameters(), lr=1)

layer1.weight.data[:] = 1
layer1.bias.data[:] = 0
layer2.weight.data[:] = 1
layer2.bias.data[:] = 0

print(*[(name, param) for name,param in layer1.named_parameters()])
print(*[(name, param) for name,param in layer2.named_parameters()])

y1_hat = layer1(x)
y2_hat = layer2(y1_hat)
y1 = th.tensor(1)
y2 = th.tensor(1)
loss1 = (y1_hat.mean()-y1) ** 2
loss2 = (y2_hat-y2) ** 2

print(layer1.weight.grad)
print(layer2.weight.grad)

opt1.zero_grad()
opt2.zero_grad()
num = 2
if num == 1:
    loss2.backward()
else:
    (loss1 + loss2).backward()

print(layer1.weight.grad)
print(layer2.weight.grad)

# 对layer2更新参数
opt2.step()
print(1)
print(*[(name, param) for name,param in layer1.named_parameters()])
print(*[(name, param) for name,param in layer2.named_parameters()])

# 对layer1更新参数
opt1.step()
print(2)
print(*[(name, param) for name,param in layer1.named_parameters()])
print(*[(name, param) for name,param in layer2.named_parameters()])

'''
结论： 不同opt更新不同的net，loss1与loss2梯度会反传到layer1，用opt1更新时两种梯度都会影响更新
'''

