from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.nn import Parameter
import math
# https://blog.csdn.net/tsq292978891/article/details/79364140
grads=[]
def print_grad(tag):
    def wrapper(grad):
        print(tag,grad)
        grads.append(grad.clone())
    return wrapper
class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
        """
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def predict(self,input):
        return F.linear(F.normalize(input), F.normalize(self.weight))

    def forward(self, input,label):
        self.one_hot = torch.zeros((label.size(0), self.out_features), dtype=torch.uint8)
        self.one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        # cosine.register_hook(print_grad('cos(t)'))
        sine = torch.sqrt((1.000001 - torch.pow(cosine, 2)) )
        phi = cosine * self.cos_m - sine * self.sin_m

        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        self.output = torch.where(self.one_hot,phi,cosine)
        # self.output.register_hook(print_grad('cos(t+m)'))
        return self.output*self.s



    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', m=' + str(self.m) + ')'

# https://pytorch.org/docs/stable/notes/extending.html#extending-torch-autograd

class ArcMarginLoss(nn.Module):
    def __init__(self,in_feature,out_feature,m=0.5,s=30):
        super().__init__()
        self.in_feature=in_feature
        self.out_feature=out_feature
        self.m=torch.Tensor([m])
        self.s=torch.Tensor([s])
        self.weight = nn.Parameter(torch.FloatTensor(out_feature,in_feature))
        nn.init.kaiming_normal_(self.weight)


    def forward(self, input,labels):
        return ArcMarginLossFun.apply(F.normalize(input), labels,
                          F.normalize(self.weight),
                          self.m,self.s)

    def predict(self,input):
        return F.linear(input,self.weight)

class ArcMarginLossFun(Function):
    @staticmethod
    # bias is an optional argument
    def forward(ctx, ninput, labels, nW, m,s):
        cos_m, sin_m = m.cos(), m.sin()
        cosine = F.linear(ninput,nW)
        # cosine = F.linear(input, W)
        sine = (1.0 - torch.pow(cosine, 2)).clamp_(0, 1)
        sine.sqrt_()
        # 和差化积公式: phi = cos(theta+m)
        phi = cosine * cos_m - sine * sin_m

        one_hot = torch.zeros((ninput.size(0),nW.size(0)))
        one_hot.scatter_(1,labels.reshape(-1,1),1)

        phi=phi.where(one_hot.type(torch.uint8),cosine)
        z = F.softmax(phi*s,dim=1)
        ctx.save_for_backward(ninput, nW, cosine,sine,phi,one_hot, z,m,s)

        return -(one_hot*z.log()).sum()/ninput.size(0)

    # 返回的梯度个数与输入个数一致
    @staticmethod
    def backward(ctx, grad_output):
        input, W, cosine, sine, phi,one_hot,z, m, s = ctx.saved_tensors
        cos_m, sin_m = m.cos(), m.sin()
        gz = (z-one_hot)*s/input.size(0)

        dphi = (cosine*sin_m+sine*cos_m)/(sine+1e-6)
        gz=gz.where(~one_hot.type(torch.uint8),gz*dphi)
        # print('sin', sine)
#input.pow(2).sum(0).sqrt_()
        nx =gz.mm(W)
        # print('nx',nx)
        return  nx, None, \
               gz.t().mm(input),None,None

def main():
    pass

if __name__ == '__main__':
    loss_fun1 = ArcMarginProduct(2, 4)
    loss_fun2 = ArcMarginLoss(2, 4)
    x1 = torch.Tensor([[1,1]])
    x1.requires_grad_()
    loss_fun1.weight.data= torch.Tensor([
        [1,0],
        [0,1],
        [-1,0],
        [0,-1]
    ])
    labels = torch.arange(len(x1))%4


    def loss_fun(x, labels):
        return F.cross_entropy(loss_fun1(x, labels),labels)
    opt=torch.optim.Adam([loss_fun1.weight])

    for i in range(1000):
        loss1 = loss_fun(x1, labels)
        opt.zero_grad()
        loss1.backward()
        opt.step()

    # grad_x = torch.zeros_like(x1)
    # for i in range(x1.size(0)):
    #     for j in range(x1.size(1)):
    #         x = x1.clone()
    #         d = 1e-6
    #         x[i, j] += d
    #         loss = loss_fun(x, labels)
    #         grad_x[i, j] = (loss - loss1) / d
    #
    # grad_w = torch.zeros_like(loss_fun1.weight)
    # w = loss_fun1.weight.clone()
    # for i in range(loss_fun1.weight.size(0)):
    #     for j in range(loss_fun1.weight.size(1)):
    #         d = 1e-3
    #         loss_fun1.weight.data[i, j] += d
    #         loss = loss_fun(x1, labels)
    #         grad_w.data[i, j] = (loss - loss1) / d
    #         loss_fun1.weight.data = w.clone()

    # print('gx',x1.grad / grad_x)
    # print('gw',loss_fun1.weight.grad - grad_w)



