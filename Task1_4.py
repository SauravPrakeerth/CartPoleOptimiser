from cgi import test
from pyexpat import model
from random import choice
from Task1_1 import rollout

from cv2 import INTERSECT_PARTIAL
from CartPole import CartPole, remap_angle
import random
import numpy
import autograd.numpy as np
from matplotlib.pyplot import ion, draw, Rectangle, Line2D
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

trainObject = CartPole()

Xdata, Ydata_CL, Ydata_CV, Ydata_PA, Ydata_PV = [], [], [], [], []

for i in range(500):
    X = [random.uniform(-5,5),random.uniform(-10,10),random.uniform(-np.pi,np.pi),random.uniform(-15,15)] # intial state
    trainObject.setState(X)
    trainObject.performAction()
    Y = np.array(trainObject.getState()) - np.array(X)
    Xdata.append(X)
    Ydata_CL.append(Y[0])
    Ydata_CV.append(Y[1])
    Ydata_PA.append(Y[2])
    Ydata_PV.append(Y[3])
    trainObject.reset()

Xdata = (np.array(Xdata)).reshape(-1,4)
model_CL = LinearRegression().fit(Xdata, Ydata_CL)
model_CV = LinearRegression().fit(Xdata, Ydata_CV)
model_PA = LinearRegression().fit(Xdata, Ydata_PA)
model_PV = LinearRegression().fit(Xdata, Ydata_PV)


# Coefficient of determination
print(f' CL R^2: {model_CL.score(Xdata, Ydata_CL)}')
print(f' CV R^2: {model_CV.score(Xdata, Ydata_CV)}') # not predicated well
print(f' PA R^2: {model_PA.score(Xdata, Ydata_PA)}') 
print(f' PV R^2: {model_PV.score(Xdata, Ydata_PV)}') # not predicted well


coefficient_matrix = np.array([list(model_CL.coef_),list(model_CV.coef_),list(model_PA.coef_),list(model_PV.coef_)])
print(coefficient_matrix)

#### SIMPLE OSCILLATION ABOUT STABLE EQUILIBRIUM ####

trainedObject = CartPole()
# initial_state = [0,0,0,6] ## CASE 1
# initial_state = [0,0,3,0]  ## CASE 2
# initial_state = [0,0,np.pi,15]  ## CASE 4
initial_state = [2,-2,0,-2]  ## CASE 5
num_time_steps = 10
trainedObject.setState(initial_state)
state_array = [initial_state]

curr_state = np.array(initial_state)
for t in range(num_time_steps-1):
    curr_change = np.matmul(coefficient_matrix, np.transpose(curr_state))
    next_state = curr_state+curr_change
    next_state[2] = remap_angle(next_state[2])
    state_array.append(next_state)
    curr_state = next_state

testObject = CartPole()
test_CL, test_CV, test_PA, test_PV, time_array = rollout(testObject,initial_state,num_time_steps)

#### PLOTTING ####

plt.subplot(2, 2, 1)
plt.plot(time_array, test_CL, label = "Real")
plt.plot(time_array, [state_array[i][0] for i in range(np.shape(state_array)[0])], label = "Predicted")
plt.legend()
plt.title("Cart Location")
plt.xlabel('Time')
plt.ylabel('X')

plt.subplot(2, 2, 2)
plt.plot(time_array, test_CV, label = "Real")
plt.plot(time_array, [state_array[i][1] for i in range(np.shape(state_array)[0])], label = "Predicted")
plt.legend()
plt.title("Cart Velocity")
plt.xlabel('Time')
plt.ylabel('X')

plt.subplot(2, 2, 3)
plt.plot(time_array, test_PA, label = "Real")
plt.plot(time_array, [state_array[i][2] for i in range(np.shape(state_array)[0])], label = "Predicted")
plt.legend()
plt.title("Pole Angle")
plt.xlabel('Time')
plt.ylabel('X')

plt.subplot(2, 2, 4)
plt.plot(time_array, test_PV, label = "Real")
plt.plot(time_array, [state_array[i][3] for i in range(np.shape(state_array)[0])], label = "Predicted")
plt.legend()
plt.title("Pole Velocity")
plt.xlabel('Time')
plt.ylabel('X')

plt.show()