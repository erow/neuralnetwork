# https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html#sphx-glr-beginner-blitz-autograd-tutorial-py

import torch as T
import torch.nn.functional as F
import matplotlib.pyplot as plt

x = T.Tensor([1, 1])
x = T.nn.Parameter(x)
_x = x.clone()
target = T.Tensor([1, 0])


def print_grad(tag):
    def wrapper(grad):
        print(tag, grad)

    return wrapper


# x.register_hook(print_grad('x'))

# argmax nx * x-axis= cos(t),  x不断向x轴靠拢

#
x.register_hook(print_grad('x'))
nx = F.normalize(x, dim=0)
loss = (nx * target).sum()
nx.register_hook(print_grad('nx'))
loss.backward()

# lr = 1e-2
# opt = T.optim.SGD([x], lr)
# # 使用自动求导
# ts = []
# for i in range(300):
#     theta = T.atan2(x[1], x[0]).item()
#     ts.append(theta)
#     print('theta:', theta)
#     nx = F.normalize(x, dim=0)
#     loss = -(nx * target).sum()
#     # opt.zero_grad()
#     # loss.backward()
#     # opt.step()
#     x.grad = None
#     loss.backward()
#     x.data -= x.grad * lr
#
# x.data = _x
# ts1 = []
# for i in range(300):
#     theta = T.atan2(x[1], x[0]).item()
#     ts1.append(theta)
#     print('theta:', theta)
#     x.data = F.normalize(x, dim=0)
#     loss = -(x * target).sum()
#     # opt.zero_grad()
#     # loss.backward()
#     # opt.step()
#     x.grad = None
#     loss.backward()
#     x.data -= x.grad * lr
# plt.subplots(2, 1)
# plt.subplot(211)
# plt.plot(ts)
# plt.plot(ts1)
# plt.subplot(212)
# plt.plot([(ts[i + 1] + 1e-6) / (1e-6 + ts[i]) for i in range(299)])
# # 说明梯度下降实际上是角度的指数下降
# plt.show()
