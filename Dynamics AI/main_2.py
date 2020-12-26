import numpy as np
import math, random
import matplotlib.pyplot as plt
import tensorflow as tf
import casadi
from casadi import MX

def plant_dynamics_simple(x,u):
    return 1*x+4*u

def plant_dynamics_damaged_simple(x,u):
    return 1*x+3*u+math.tanh(x*0.003)*10+u*5*math.sin(x*0.2)

def main():
    tf.random.set_seed(42)
    samples = 1000
    #Input is u and x, train with u's and x's go to actual values
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(units=2, input_shape=(2,)),
        tf.keras.layers.Dense(units=samples*(2+1), activation='relu'),
        tf.keras.layers.Dense(1, activation='linear'),
    ])
    tarray = range(0,samples)
    xarray = []
    uarray = []
    errorArray = []
    for t in tarray:
        x = random.random()*100
        u = random.random()*100
        xarray.append(x)
        uarray.append(u)
        actual_dynamics = plant_dynamics_damaged_simple(x,u)
        expected_dynamics = plant_dynamics_simple(x,u)
        error = actual_dynamics-expected_dynamics
        errorArray.append(error)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer,
              loss='mae',
              metrics=['accuracy'])
    print(model.summary())
    model.fit(np.array([errorArray,uarray]).T, np.array(errorArray), epochs=60)

    #Test data with a "real loop"
    x = 0
    u = 0
    errorArray = []
    uarray = []
    tarray = range(0,80)
    for t in tarray:
        actual_dynamics = plant_dynamics_damaged_simple(x,u)
        expected_dynamics = plant_dynamics_simple(x,u)
        errorArray.append(actual_dynamics-expected_dynamics)
        uarray.append(u)
        x = actual_dynamics
        u = math.sin(t/80)
    prediction = model.predict(np.array([errorArray,uarray]).T)

    #Test that this model will correctly correct for the damaged plant
    plt.plot(tarray, errorArray)
    plt.plot(tarray, prediction)
    plt.legend()
    plt.show()

def mpc():
    N = 3 #horizon length
    wu = 0.1
    wx = 1
    r = 5 #setpoint
    xs = []
    dus = []
    for i in range(0,N):
        xs.append(MX.sym("x"+str(i)))
        dus.append(MX.sym("du"+str(i)))
    J = 0
    for i in range(0,N):
        J += wx*(r-xs[i])**2+wu*dus[i]**2
    qp = {'x':casadi.vertcat(xs[0],xs[1],xs[2],dus[0],dus[1],dus[2]), 'f':J, 'g':xs[0]-50}
    S = casadi.qpsol('S', 'qpoases', qp)
    print(S)
    r = S(lbg=0)
    x_opt = r['x']
    print('x_opt: ', x_opt)

if __name__ == "__main__":
    mpc()