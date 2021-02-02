from numpy.lib.function_base import append
import torch
from torch import nn, optim
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import math,random
from torch.tensor import Tensor

def f(x):
    return x**2+0.5*x+1

trainFeatures = []
testFeatures = []
trainLabels = []
testLabels = []
for i in range(0,1000):
    trainFeatures.append(100*random.random())
    testFeatures.append(100*random.random())
    trainLabels.append(f(trainFeatures[-1]))
    testLabels.append(f(testFeatures[-1]))

model = nn.Sequential(
    nn.Linear(1,100),
    nn.ReLU(),
    nn.Linear(100,100),
    nn.ReLU(),
    nn.Linear(100,1)
)

optimizer = optim.Adam(model.parameters(), lr=0.005)
criterion = nn.MSELoss()

def train():
    model.train()
    losses = []
    for epoch in range(1,2000):
        x_train = Variable(torch.from_numpy(np.array([trainFeatures]).T)).float()
        y_train = Variable(torch.from_numpy(np.array([trainLabels]).T)).float()
        y_pred = model(x_train)
        loss = criterion(y_pred, y_train)
        # print ("epoch #", epoch)
        # print (loss.item())
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return losses
losses = train()
# print ("testing start ... ")
x_test = Variable(torch.from_numpy(np.array([testFeatures])).T).float()
x_train = Variable(torch.from_numpy(np.array([testLabels])).T).float()
print(testFeatures)
for i in range(0,len(testLabels)):
    print(testLabels[i], "    ", model(x_test[i]).data.numpy())
xs = []
yAct = []
yPred = []
for x in np.linspace(0,20,100):
    xs.append(float(x))
    yAct.append(f(x))
x_test2 = Variable(torch.from_numpy(np.array([xs])).T).float()
yPred = model(x_test2).data.numpy()


# axs[0].xlabel("epoch")
# axs[0].ylabel("loss train")
axs[1].plot(xs,yAct)
axs[1].plot(xs,yPred)
plt.show()



# def test():
#     pred = net(x_test)
#     pred = torch.max(pred, 1)[1]
#     print ("Accuracy on test set: ", accuracy_score(labels_test, pred.data.numpy()))

#     p_train = net(x_train)
#     p_train = torch.max(p_train, 1)[1]
#     print ("Accuracy on train set: ", accuracy_score(labels_train, p_train.data.numpy()))
           
# test()