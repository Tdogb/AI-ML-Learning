from pyfme.models.state import angular_velocity
from pyfme.utils import input_generator
from pyfme.utils.input_generator import vectorize_float
from numpy.lib.function_base import append
import torch
from torch import nn, optim
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import math,random
from torch.tensor import Tensor

'''
height system.full_state.position.height

psi system.full_state.attitude.psi
theta system.full_state.attitude.theta
phi system.full_state.attitude.phi

p system.full_state.angular_vel.p
q system.full_state.angular_vel.q
r system.full_state.angular_vel.r

returns a an array of [elevator, aileron, rudder, dt]
'''
class ControlFunction():
    trainFeatures = np.array([[]])
    trainLabels = np.array([[]])
    def __init__(self):
        pass

    def __call__(self, state):
        # print(state)
        return [0.1,0,0,0.1]

    def Log(state):

