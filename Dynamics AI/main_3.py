import os, sys
import numpy as np
import math, random
import matplotlib.pyplot as plt
import tensorflow as tf
import casadi
from casadi import MX

def plant_dynamics_simple(x,u):
    return 1*x+4*u

def plant_dynamics_damaged_simple(x,u):
    return 1*x+3*u+u*9*math.sin(x*0.05)#+math.tanh(x*0.003)*10 + 1

def nn(xi):
    # tf.random.set_seed(4305398252)
    samples = 1000
    #Input is u and x, train with u's and x's go to actual values
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(units=1, input_shape=(2,)),
        tf.keras.layers.Dense(units=samples*(2+1), activation='relu'),
        tf.keras.layers.Dense(1, activation='linear'),
    ])
    tarray = range(0,samples)
    xarray = []
    uarray = []
    errorArray = []
    for t in tarray:
        u = random.random()*8
        x = xi if len(xarray) == 0 else xarray[-1]
        actual_dynamics = plant_dynamics_damaged_simple(x,u)
        expected_dynamics = plant_dynamics_simple(x,u)
        xarray.append(actual_dynamics)
        uarray.append(u)
        error = actual_dynamics-expected_dynamics
        errorArray.append(error)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer,
              loss='mae',
              metrics=['accuracy'])
    print(model.summary())
    model.fit(np.array([xarray,uarray]).T, np.array(errorArray).T, epochs=90)
    return model, xarray, uarray
    #Test data with a "real loop"
    x = xi
    u = 0
    errorArray = []
    uarray = []
    xarray = []
    tarray = range(0,80)
    for t in tarray:
        actual_dynamics = plant_dynamics_damaged_simple(x,u)
        expected_dynamics = plant_dynamics_simple(x,u)
        xarray.append(actual_dynamics)
        errorArray.append(actual_dynamics-expected_dynamics)
        uarray.append(u)
        x = actual_dynamics
        u = 2*math.sin(float(t)/80)
    prediction = model.predict(np.array([xarray,uarray]).T)
    print(prediction)
    #Test that this model will correctly correct for the damaged plant
    ePlot = plt.plot(tarray, errorArray)
    pPlot = plt.plot(tarray, prediction)
    plt.legend(("actual","prediction"))
    plt.show()


def mpc(xi, setpoint):
    opti = casadi.Opti()
    print("Starting MPC")
    N = 10 #horizon length
    wu = 0.1 #weight of control effort (u)
    wx = 1 #weight of x's
    du_bounds = 0.05
    r = setpoint #setpoint
    xs = []
    dus = []
    for i in range(0,N):
        xs.append(opti.variable())
        dus.append(opti.variable())
        #Actuator effort constraints
        opti.subject_to(dus[-1] >= -du_bounds)
        opti.subject_to(dus[-1] <= du_bounds)
        #Dynamics constraint. Nessesary to have a different first constraint since the first x isn't a decsision variable
        #Dynamics are copied/pasted from the plant_dynamics_damaged_simple function since the casadi solver cannot have function calls in it
        if i == 0:
            opti.subject_to(xs[-1] - (1*xi+4*dus[-1]) == 0)
            dus.append(opti.variable())
            opti.subject_to(dus[-1] >= -du_bounds)
            opti.subject_to(dus[-1] <= du_bounds)
        else:
            opti.subject_to(xs[-1] - (1*xs[-2]+4*dus[-2]) == 0)
    #Cost function for MPC taken from https://en.wikipedia.org/wiki/Model_predictive_control
    J = wx*(r-xi)**2+wu*dus[0]**2
    for i in range(1,N):
        J += wx*(r-xs[i-1])**2+wu*dus[i]**2
    # sys.stdout = open(os.devnull, 'w')
    opti.minimize(J)
    opti.solver('ipopt')
    sol = opti.solve()
    # sys.stdout = sys.__stdout__
    return sol.value(dus[0])
    print("xs " + str(xi) + " dus: " + str(sol.value(dus[0])))
    for i in range(0, len(xs)):
        print("xs " + str(sol.value(xs[i])) + " dus: " + str(sol.value(dus[i+1])))
def simulate():
    timesteps = 60
    error_threshold = 0.2
    setpoint = 20
    error_array = []
    model_correction_array = []
    u = [0]
    x = [plant_dynamics_simple(0,u[-1])]
    damaged = False
    model = None
    for t in range(0,timesteps):
        damaged = t > timesteps/4
        x_expected = plant_dynamics_simple(x[-1],u[-1])
        if damaged:
            #xi+1
            x.append(plant_dynamics_damaged_simple(x[-1],u[-1]))
        else:
            x.append(plant_dynamics_simple(x[-1],u[-1]))
        error = x[-1] - x_expected
        print("error " + str(error) + " x " + str(x[-1]))
        error_array.append(error)
        if  error > error_threshold:
            model, xarray, uarray = nn(x[-1])
        model_correction = 0
        if model != None:
            model_correction = model.predict(np.array([[x[-1]],[u[-1]]]).T)
            model_correction_array.append(model_correction[0,0])
            u.append(mpc(x[-1], setpoint) + model_correction[0,0])
        else:
            u.append(mpc(x[-1], setpoint))
    actual_plant_output = [x[0]]
    for u_ in u:
        actual_plant_output.append(plant_dynamics_damaged_simple(actual_plant_output[-1],u_))
    fig,axs = plt.subplots(2,2)
    axs[0,0].plot(actual_plant_output)
    axs[0,1].plot(error_array)
    axs[1,0].plot(model_correction_array)
    axs[0,0].set_title("Actual Plant Output")
    axs[0,1].set_title("Error")
    axs[1,0].set_title("Model Correction")
    plt.show()

def information_gain():
    opti = casadi.Opti()
    B = 0.5
    ui = 0
    u = [] #This is the initial training point
    N = len(u)+1 #The amount of previous training data (so we see if we are actually learning anything new). This is not usually a constant
    I = 0
    for i in range(0,N):
        u.append(opti.variable()) #uk
        if i != 0:
            I += casadi.norm_1(u[-1]-u[-2])
        else:
            I += casadi.norm_1(u[-1]-ui)
    #g(u)
if __name__ == "__main__":
    simulate()