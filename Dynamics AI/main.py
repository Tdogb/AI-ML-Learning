import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math

def plant_dynamics(x, u):
    A = np.array([[1.5,0],[0,1.2]])
    B = np.array([1,1]).T
    return A@x + B@u

def plant_dynamics_damaged(x, u):
    A = np.array([[1.5-1,0],[0,1.2]])
    B = np.array([1,1]).T
    return A@x + B@u

def plant_dynamics_simple(x,u):
    return 1*x+4*u

def plant_dynamics_damaged_simple(x,u):
    return 1*x+3*u+math.tanh(x*0.003)*10# #+ u*5*math.sin(x*0.2)
#This will be the controller for the normal plant dynamics
def plant_inverse_dynamics(x_new, x):
    return 0

#safe trajectory
def xr(t):
    return np.array([t * 3.1, t/3]).T

def main():
    tf.random.set_seed(42)
    samples = 20
    #Input is u and x, train with u's and x's go to actual values
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(units=2, input_shape=(2,)),
        tf.keras.layers.Dense(units=40*(2+1), activation='relu'),
        tf.keras.layers.Dense(1, activation='linear'),
    ])
    x = 0
    error = 0
    error_array = []
    x_array = []
    u_array = []
    x_p1_array = [] #actual dynamics array
    x_p1_e_array = [] #expected dynamics array
    t_array = range(0,samples)
    for t in t_array: #Arbitrary amount of points in my simple example
        u = t*0.4
        actual_dynamics = plant_dynamics_damaged_simple(x,u)
        expected_dynamics = plant_dynamics_simple(x,u)
        error = actual_dynamics-expected_dynamics
        error_array.append(error)
        x_array.append(x)
        u_array.append(u)
        x_p1_array.append(actual_dynamics)
        x_p1_e_array.append(expected_dynamics)
        x = actual_dynamics
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    model.compile(optimizer=optimizer,
              loss='mae',
              metrics=['accuracy'])
    print(model.summary())
    print(np.array([error_array,u_array]).T)
    model.fit(np.array([error_array,u_array]).T, np.array(error_array), epochs=500)
    print("Prediction")
    prediction = model.predict(np.array([error_array,u_array]).T, verbose=1)
    # prediction = np.ndarray.flatten(prediction) + x_p1_e_array
    print(prediction)
    t_array_2 = range(samples, samples+30)
    prediction2 = model.predict(np.array([error_array,u_array]).T, verbose=1)
    plt.plot(t_array, error_array)
    plt.plot(t_array, prediction)
    plt.show()
if __name__ == "__main__":
    main()