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
    opti = casadi.Opti()
    N = 4 #horizon length
    wu = 0.1 #weight of control effort (u)
    wx = 1 #weight of x's
    du_bounds = 0.5
    r = 5 #setpoint
    xi = 0
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
            opti.subject_to(xs[-1] - (1*xi+3*dus[-1]+casadi.tanh(xi*0.003)*10+dus[-1]*5*casadi.sin(xi*0.2)) == 0)
            dus.append(opti.variable())
            opti.subject_to(dus[-1] >= -du_bounds)
            opti.subject_to(dus[-1] <= du_bounds)
        else:
            opti.subject_to(xs[-1] - (1*xs[-2]+3*dus[-2]+casadi.tanh(xs[-2]*0.003)*10+dus[-2]*5*casadi.sin(xs[-2]*0.2)) == 0)
    #Cost function for MPC taken from https://en.wikipedia.org/wiki/Model_predictive_control
    J = wx*(r-xi)**2+wu*dus[0]**2
    for i in range(1,N):
        J += wx*(r-xs[i-1])**2+wu*dus[i]**2

    opti.minimize(J)
    opti.solver('ipopt')
    sol = opti.solve()
    print("xs " + str(xi) + " dus: " + str(sol.value(dus[0])))
    for i in range(0, len(xs)):
        print("xs " + str(sol.value(xs[i])) + " dus: " + str(sol.value(dus[i+1])))

if __name__ == "__main__":
    mpc()