import xpc
import sys, threading, time, math, random
from numpy.lib.function_base import append
import numpy as np
import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.tensor import Tensor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

states = [[]]
controls = [[]]

def monitor():
    with xpc.XPlaneConnect() as client:
        while True:
            posi = client.getPOSI();
            ctrl = client.getCTRL();
            if(posi != []):
                states.append(list(posi))
                controls.append(list(ctrl))
            # print(states)
            #Height, airspeed, roll, pitch, yaw, aroll, apitch, ayaw
            #Throttle, roll, pitch, yaw
            # print "Loc: (%4f, %4f, %4f) Aileron:%2f Elevator:%2f Rudder:%2f\n"\
            #    % (posi[0], posi[1], posi[2], ctrl[1], ctrl[0], ctrl[2])
            time.sleep(0.01)

model = nn.Sequential(
nn.Linear(4,100),
nn.ReLU(),
nn.Linear(100,100),
nn.ReLU(),
nn.Linear(100,8)
)
optimizer = optim.Adam(model.parameters(), lr=0.005)
criterion = nn.CrossEntropyLoss()

def nn():
    while len(states) < 50:
        print(len(states))
    losses = train()
    fig, axs = plt.subplots(2)
    axs[0].plot(range(1, 2000), losses)
    plt.show()

def train():
    model.train()
    losses = []
    x_train_t, x_test_t, y_train_t, y_test_t = train_test_split(states, controls, test_size=0.33)
    for epoch in range(1,200):
        # x_train = Variable(torch.from_numpy(np.array(x_train_t[int(random.random())*len(x_train_t)-1])).T).float()
        x_train = Variable(torch.from_numpy(np.array(x_train_t)).T).float()
        y_train = Variable(torch.from_numpy(np.array(y_train_t)).T).float()
        y_pred = model(x_train)
        loss = criterion(y_pred, y_train)
        # print ("epoch #", epoch)
        # print (loss.item())
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return losses

if __name__ == "__main__":
    monitor_thread = threading.Thread(target=monitor)
    nn_thread = threading.Thread(target=nn)
    monitor_thread.start()
    nn_thread.start()