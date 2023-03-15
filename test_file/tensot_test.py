import torch as th
a = th.randint(0, 20, (2,3,4))
b = a[:, 1, :]
print(a)
print(b)