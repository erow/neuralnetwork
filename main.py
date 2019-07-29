import matplotlib.pyplot as plt
import pandas as pd
from torch import optim

from losses import *

df=pd.read_csv('sat.trn',sep=' ',header=None)
mean = df.values[:,:-1].mean(0)
std = df.values[:,:-1].std(0)
x_trn = (df.values[:,:-1]-mean)/std
y_trn = df.values[:,-1] - 1
df=pd.read_csv('sat.tst',sep=' ',header=None)
x_tst = (df.values[:,:-1]-mean)/std
y_tst = df.values[:,-1] - 1
y_trn[y_trn==6]-=1
y_tst[y_tst==6]-=1


def accuracy(model,tx,ty):
    features=model(tx).detach()
    output=metric_fc.predict(features)
    return (output.argmax(1)==ty).float().mean()
def T(x):
    return torch.Tensor(x)
x_trn, y_trn = T(x_trn),T(y_trn).long()
x_tst, y_tst = T(x_tst), T(y_tst).long()

model = nn.Sequential(
    nn.Linear(36,100),
    nn.LeakyReLU(),
    nn.Linear(100,2),

)


criterion = nn.CrossEntropyLoss()
metric_fc=ArcMarginProduct(2,6,s=30,m=0.2)
opt1= optim.Adam(model.parameters(),1e-3,)
opt1.add_param_group({'params':metric_fc.weight})
model.train()

for epoch in range(3000):
    features=model(x_trn)
    output = metric_fc(features,y_trn)
    loss = criterion(output,y_trn)
    # loss = metric_fc(features,y_trn)
    opt1.zero_grad()
    loss.backward()
    opt1.step()
    if (epoch+1) % 200==0:
        print(loss.item(),accuracy(model,x_trn,y_trn).item())


ex=model(x_trn).detach()
for color in range(6):
    mask = (y_trn%6) == color
    plt.scatter(ex[mask.nonzero(),0],ex[mask.nonzero(),1],label=color,)
plt.legend()
plt.show()