from cgi import test
from curses import nonl
from pyexpat import model
from random import choice
from rollout2 import rollout

from cv2 import INTERSECT_PARTIAL
from CartPole import CartPole, remap_angle
import random
import numpy
import autograd.numpy as np
from matplotlib.pyplot import ion, draw, Rectangle, Line2D
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#### LINEAR MODEL WITH F ####

trainObject = CartPole()

Xdata, Ydata_CL, Ydata_CV, Ydata_PA, Ydata_PV = [], [], [], [], []

N = 10000

# GNOISE
nMean = 0
nStd = 1

for i in range(N):
    X = [random.uniform(-5,5),random.uniform(-10,10),random.uniform(-np.pi,np.pi),random.uniform(-15,15),random.uniform(-5,5)] # intial state
    trainObject.setState(X[:4], nMean, nStd)
    trainObject.performAction(X[4])
    Y = np.array(trainObject.getState()) - np.array(X[:4])
    Xdata.append(X)
    Ydata_CL.append(Y[0])
    Ydata_CV.append(Y[1])
    Ydata_PA.append(Y[2])
    Ydata_PV.append(Y[3])
    trainObject.reset()


Xdata = (np.array(Xdata)).reshape(-1,5)
model_CL = LinearRegression().fit(Xdata, Ydata_CL)
model_CV = LinearRegression().fit(Xdata, Ydata_CV)
model_PA = LinearRegression().fit(Xdata, Ydata_PA)
model_PV = LinearRegression().fit(Xdata, Ydata_PV)


# Coefficient of determination
#print(f' CL R^2: {model_CL.score(Xdata, Ydata_CL)}')
#print(f' CV R^2: {model_CV.score(Xdata, Ydata_CV)}') # not predicated well
#print(f' PA R^2: {model_PA.score(Xdata, Ydata_PA)}') 
#print(f' PV R^2: {model_PV.score(Xdata, Ydata_PV)}') # not predicted well


coefficient_matrix = np.array([list(model_CL.coef_),list(model_CV.coef_),list(model_PA.coef_),list(model_PV.coef_)])
#print(coefficient_matrix)


trainedObject = CartPole()
# initial_state = [0,0,0,6,-3] ## CASE 1
# initial_state = [0,0,3,0,-4]  ## CASE 2
## initial_state = [0,0,np.pi,15,1]  ## CASE 4
initial_state = [2,-2,0,-2,2]  ## CASE 5
num_time_steps = 10
trainedObject.setState(initial_state)
state_array = [initial_state]

curr_state = np.array(initial_state)
for t in range(num_time_steps-1):
    curr_change = np.matmul(coefficient_matrix, np.transpose(curr_state))
    next_state = curr_state[:4]+curr_change
    next_state[2] = remap_angle(next_state[2])
    next_state = list(next_state)
    next_state.append(curr_state[4])
    state_array.append(next_state)
    curr_state = next_state

testObject = CartPole()
test_CL, test_CV, test_PA, test_PV, time_array = rollout(testObject,initial_state,num_time_steps)

## PLOTTING YPRED VS Y ##
 
plt.subplot(1, 4, 1) 
plt.scatter(Ydata_CL, model_CL.predict(Xdata))
plt.title("Cart Location")
plt.xlabel('Y')
plt.ylabel('Predicted Y')

plt.subplot(1, 4, 2) 
plt.scatter(Ydata_CV, model_CV.predict(Xdata))
plt.title("Cart Velocity")
plt.xlabel('Y')
plt.ylabel('Predicted Y')

plt.subplot(1, 4, 3) 
plt.scatter(Ydata_PA, model_PA.predict(Xdata))
plt.title("Pole Angle")
plt.xlabel('Y')
plt.ylabel('Predicted Y')

plt.subplot(1, 4, 4)
plt.scatter(Ydata_PV, model_PV.predict(Xdata))
plt.title("Pole Velocity")
plt.xlabel('Y')
plt.ylabel('Predicted Y')

plt.show()

'''
## PLOTTING ROLLOUT ##

plt.subplot(1, 4, 1)
plt.plot(time_array, test_CL, label = "Real")
plt.plot(time_array, [state_array[i][0] for i in range(np.shape(state_array)[0])], label = "Predicted")
plt.legend()
plt.title("Cart Location")
plt.xlabel('Time')
plt.ylabel('X')

plt.subplot(1, 4, 2)
plt.plot(time_array, test_CV, label = "Real")
plt.plot(time_array, [state_array[i][1] for i in range(np.shape(state_array)[0])], label = "Predicted")
plt.legend()
plt.title("Cart Velocity")
plt.xlabel('Time')
plt.ylabel('X')

plt.subplot(1, 4, 3)
plt.plot(time_array, test_PA, label = "Real")
plt.plot(time_array, [state_array[i][2] for i in range(np.shape(state_array)[0])], label = "Predicted")
plt.legend()
plt.title("Pole Angle")
plt.xlabel('Time')
plt.ylabel('X')

plt.subplot(1, 4, 4)
plt.plot(time_array, test_PV, label = "Real")
plt.plot(time_array, [state_array[i][3] for i in range(np.shape(state_array)[0])], label = "Predicted")
plt.legend()
plt.title("Pole Velocity")
plt.xlabel('Time')
plt.ylabel('X')

plt.show()
'''
'''
## 1D scans ##
initial_states = [random.uniform(-5,5), random.uniform(-10,10), random.uniform(-np.pi,np.pi), random.uniform(-15,15), random.uniform(-5,5)]
Xarray = [[cart_loc for cart_loc in range(-5,5)],[cart_vel for cart_vel in range(-10,10)],[pole_ang for pole_ang in numpy.linspace(-np.pi,np.pi,30)], [pole_vel for pole_vel in range(-15,15)], [force for force in range(-5,5)]]
labels = ['CL', 'CV', 'PA', 'PV', 'F']

def pred(state, y_ind):
    print(state)
    if y_ind == 0:
        return model_CL.predict([state])
    elif y_ind == 1:
        return model_CV.predict([state])
    elif y_ind == 2:
        return model_PA.predict([state])
    elif y_ind == 3:
        return model_PV.predict([state])

def scan(sx,sy):
    x_array = []
    y_array = []
    for x in Xarray[sx]:
        state = initial_states
        state[sx] = x
        x_array.append(x)
        y_pred = pred(state, sy)
        y_array.append(y_pred)
    return x_array, y_array

def plot1D(test_xy, shape):

    for i,v in enumerate(test_xy):
        plt.rcParams.update({'font.size': 5})
        plt.subplot(*shape,(i+1))
        x,y = scan(*v)
        plt.plot(x,y)
        plt.title(f'{labels[v[0]]} against {labels[v[1]]}')
    plt.show()

test_xy = [(0,0),(0,1),(0,2),(0,3),(1,0),(1,1),(1,2),(1,3),(2,0),(2,1),(2,2),(2,3),(3,0),(3,1),(3,2),(3,3),(4,0),(4,1),(4,2),(4,3)]

plot1D(test_xy, (4,5))
'''
'''
## CONTOUR PLOTS ##

initial_states = [random.uniform(-5,5), random.uniform(-10,10), random.uniform(-np.pi,np.pi), random.uniform(-15,15), random.uniform(-5,5)]
Xarray = [[cart_loc for cart_loc in range(-5,5)],[cart_vel for cart_vel in range(-10,10)],[pole_ang for pole_ang in numpy.linspace(-np.pi,np.pi,30)], [pole_vel for pole_vel in range(-15,15)], [force for force in range(-5,5)]]
labels = ['CL', 'CV', 'PA', 'PV', 'F']

plotObject = CartPole()

def contour(sx,sy,sz):
    X, Y = np.meshgrid(Xarray[sx], Xarray[sy])
    x, y = np.ravel(X), np.ravel(Y)
    z = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        state = np.array([initial_states])
        state[0][sx], state[0][sy] = x[i], y[i]
        Ypred = [model_CL.predict(state),model_CV.predict(state),model_PA.predict(state),model_PV.predict(state)]
        z[i] = Ypred[sz]
    return x,y,z

def contourPlot(test_xyz, shape):

    for i,v in enumerate(test_xyz):
        plt.rcParams.update({'font.size': 5})
        plt.subplot(*shape,(i+1))
        x,y,z = contour(*v)
        plt.tricontourf(x,y,z,levels=12)
        plt.title(f'{labels[v[2]]} = f({labels[v[0]]}, {labels[v[1]]})')
    plt.show()

test_xyz = [(0,1,0),(0,1,1),(0,1,2),(0,1,3), (0,2,0),(0,2,1),(0,2,2),(0,2,3), (0,3,0),(0,3,1),(0,3,2),(0,3,3), (0,4,0),(0,4,1),(0,4,2),(0,4,3), (1,2,0),(1,2,1),(1,2,2),(1,2,3), (1,3,0),(1,3,1),(1,3,2),(1,3,3), (1,4,0),(1,4,1),(1,4,2),(1,4,3), (2,3,0),(2,3,1),(2,3,2),(2,3,3), (2,4,0),(2,4,1),(2,4,2),(2,4,3), (3,4,0),(3,4,1),(3,4,2),(3,4,3)]

contourPlot(test_xyz, (4,10))
'''