import tensorflow as tf
import numpy as np

def plant_dynamics(x, u):
    A = np.array([[1.5,0],[0,1.2]])
    B = np.array([1,1]).T
    return A@x + B@u

def plant_dynamics_damaged(x, u):
    A = np.array([[1.5-1,0],[0,1.2]])
    B = np.array([1,1]).T
    return A@x + B@u

#This will be the controller for the normal plant dynamics
def plant_inverse_dynamics(x_new, x):
    return 0

#safe trajectory
def xr(t):
    return np.array([t * 3.1, t/3]).T

def main():
    tf.random.set_seed(42)
    duration = 10
    t = 0
    threshold = 0.0001
    x = np.array([0,0]).T
    error = 0
    u = np.array([0,0]).T
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(units=1, input_shape=(1,)),
        # tf.keras.layers.Dense(1, activation='linear'),
    ])
    x = np.array([[0]])
    y = np.array([[0]])
    y_expected = np.array([[0]])
    for t in range(0, 10): #Arbitrary amount of points in my simple example
        input = np.array([t,t]) #Our input to plant for sample gathering is arbitrarily [t,t].T
        x = np.vstack((x,input[0]))
        plant_actual = plant_dynamics_damaged([y[t].T, 0], input.T).T
        plant_expected = plant_dynamics([y[t].T, 0], input.T).T
        y = np.vstack((y, plant_actual[0]))
        y_expected = np.vstack((y_expected, plant_expected[0]))
    # print(y)
    # print(y_expected)
    model.compile(optimizer='adam',
              loss='mse',
              metrics=['accuracy'])
    print(model.summary())
    model.fit(y, y_expected, epochs=50)
    print("Prediction")
    model.predict(y, verbose=1)

    # while True:
    #     trajectory_point = xr(t)
    #     if error > threshold:

    #     else:
    #         u = plant_inverse_dynamics(trajectory_point, x)
    #     x_new = plant_dynamics(x, u)
    #     error = x_new - trajectory_point
    #     x = x_new

# def function():
#     uk = 0
#     sum = 0
#     for k in range(t,t+T):
#         sum += information_function(uk) + lamb*safety_function()

if __name__ == "__main__":
    main()