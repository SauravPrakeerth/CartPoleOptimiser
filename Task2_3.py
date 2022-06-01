from cProfile import label
from cgi import test
from curses import nonl
from pyexpat import model
from random import choice
from re import I
from telnetlib import X3PAD

from cv2 import INTERSECT_PARTIAL
from CartPole import CartPole, remap_angle
import random
import numpy
import autograd.numpy as np
from matplotlib.pyplot import ion, draw, Rectangle, Line2D
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from CartPole import CartPole, remap_angle, loss
from Task2_2 import alphaGenerator, nonlinearPred
from scipy.optimize import minimize
from numpy.random import rand
#import numpy as np

#### POLICY BASED ON REAL DYNAMICS ####


p = [random.uniform(-10,10),random.uniform(-10,10),random.uniform(-10,10),random.uniform(-10,10)]
initial_cond = np.array([0.01,0.01,0.01,0.01])
num_time_steps = 45
F = np.dot(p,initial_cond)

def rollout3(Object,initial_state,num_time_steps, p):

    Object.setState(initial_state)

    time_array = []
    CP_array = []
    CV_array = []
    PA_array = []
    PV_array = []
    X = initial_state
    loss_array = []

    for i in range(num_time_steps):
        loss_array.append(loss(X))
        Object.performAction(np.dot(p,X))
        time_array.append(i)
        X = Object.getState()
        # X = [i[0] for i in X]
        CP_array.append(X[0])
        CV_array.append(X[1])
        PA_array.append(remap_angle(X[2]))
        PV_array.append(X[3])
    loss_av = np.mean(loss_array)
    
    return time_array, loss_array, loss_av, CP_array, CV_array, PA_array, CP_array

# PLOTTING ROLLOUT #
testObject = CartPole(visual=False)

'''
## 1D SCANS ##

states = ['CL', 'CV', 'PA', 'PV']
p = [random.uniform(-10,10),random.uniform(-10,10),random.uniform(-10,10),random.uniform(-10,10)]
for ind,i in enumerate(np.linspace(-10,10,10)):
    plt.rcParams.update({'font.size': 5})
    plt.subplot(2,5,ind+1)
    for j in range(4):
        p[j] = i
        time, loss_ar, loss_av, a1, a2, a3, a4 = rollout3(testObject, initial_cond, num_time_steps,p)
        plt.plot(time, loss_ar, label = states[j])
        plt.legend()
        plt.title(f"p[state_variable] = {i}")
plt.show()
'''
## 2D SCANS ##

initial_p = [random.uniform(-10,10), random.uniform(-10,10), random.uniform(-10,10), random.uniform(-10,10)]
Xarray = [[p for p in range(-10,10)],[p for p in range(-10,10)],[p for p in range(-10,10)], [p for p in range(-10,10)]]
labels = ['p[0]', 'p[1]', 'p[2]', 'p[3]']
labelsOut = ['CL', 'CV', 'PA', 'PV']

alpha, basis, len_param, Xdata, Ydata_CL, Ydata_CV, Ydata_PA, Ydata_PV = alphaGenerator(1000,500,10**(-3))

plotObject = CartPole()

def contour(sx,sy,sz):
    X, Y = np.meshgrid(Xarray[sx], Xarray[sy])
    x, y = np.ravel(X), np.ravel(Y)
    z = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        p = np.array([initial_p])
        p[0][sx], p[0][sy] = x[i], y[i]
        time_array, loss_array, loss_av = rollout3(testObject,initial_cond,num_time_steps,p)
        z[i] = loss_av
    return x,y,z

def contourPlot(test_xyz, shape):

    for i,v in enumerate(test_xyz):
        plt.rcParams.update({'font.size': 5})
        plt.subplot(*shape,(i+1))
        x,y,z = contour(*v)
        plt.tricontourf(x,y,z,levels=12)
        plt.title(f'{labelsOut[v[2]]} = f({labels[v[0]]}, {labels[v[1]]})')
    plt.show()

test_xyz = [(0,1,0),(0,1,1),(0,1,2),(0,1,3), (0,2,0),(0,2,1),(0,2,2),(0,2,3), (0,3,0),(0,3,1),(0,3,2),(0,3,3), (1,2,0),(1,2,1),(1,2,2),(1,2,3), (1,3,0),(1,3,1),(1,3,2),(1,3,3), (2,3,0),(2,3,1),(2,3,2),(2,3,3)]

#contourPlot(test_xyz, (4,6))

def objective(p):
    initial_cond = [0,0,0.01,0]
    num_time_steps = 6
    Object = CartPole(visual=False)
    Object.setState(initial_cond)
    X = initial_cond
    loss_array = [loss(X)]

    for i in range(num_time_steps):
        Object.performAction(p[0]*X[0] + p[1]*X[1] + p[2]*X[2] + p[3]*X[3])
        X = Object.getState()
        loss_array.append(loss(X))
    loss_av = np.mean(loss_array)
    return loss_av

sol_array = []
for i in range(5):
    pt = [random.uniform(-10,10), random.uniform(-10,10), random.uniform(-10,10), random.uniform(-10,10)]
    result = minimize(objective, pt, method='Nelder-Mead')
    solution = result.x
    sol_array.append(solution)
    ''''''
'''
# plotting loss of optimised p_array

for i,p in enumerate(sol_array):
    time, loss_ar, loss_av, CP_array, CV_array, PA_array, PV_array = rollout3(testObject, initial_cond, num_time_steps,p)
    # plt.rcParams.update({'font.size': 5})
    plt.subplot(1,5,i+1)
    plt.plot(time, loss_ar)
    p_round = [round(val,2) for val in p]
    plt.title(f"p = {p_round}")
plt.show()

for i,p in enumerate(sol_array):
    time_array, loss_array, loss_av, CP_array, CV_array, PA_array, PV_array = rollout3(testObject,initial_cond,num_time_steps, p)
    plt.rcParams.update({'font.size': 5})

    plt.subplot(1,5,i+1)
    plt.plot(time_array, CP_array, label='CP')
    plt.plot(time_array, CV_array, label='CV')
    plt.plot(time_array, PA_array, label='PA')
    plt.plot(time_array, PV_array, label='PV')
    plt.legend()
    p_round = [round(val,2) for val in p]
    plt.title(f"p = {p_round}")

plt.show()
'''
'''
#### FOUND SOLUTIONS ####
sol_array = [[0.12502083,  0.19804898, 12.91606125,  1.76357623], [ 0.37944578,  0.37171809, 13.86560915,  1.98230049]]

for i,p in enumerate(sol_array):
    time, loss_ar, loss_av, CP_array, CV_array, PA_array, PV_array = rollout3(testObject, initial_cond, num_time_steps,p)
    # plt.rcParams.update({'font.size': 5})
    plt.subplot(1,2,i+1)
    plt.plot(time, loss_ar)
    p_round = [round(val,2) for val in p]
    plt.xlabel('Time')
    plt.ylabel('Loss')
    plt.title(f"p = {p_round}")
plt.show()

for i,p in enumerate(sol_array):
    time_array, loss_array, loss_av, CP_array, CV_array, PA_array, PV_array = rollout3(testObject,initial_cond,num_time_steps, p)
    #plt.rcParams.update({'font.size': 5})

    plt.subplot(1,2,i+1)
    plt.plot(time_array, CP_array, label='CP')
    plt.plot(time_array, CV_array, label='CV')
    plt.plot(time_array, PA_array, label='PA')
    plt.plot(time_array, PV_array, label='PV')
    plt.legend()
    p_round = [round(val,2) for val in p]
    plt.xlabel('Time')
    plt.ylabel('State Variables')
    plt.title(f"p = {p_round}")

plt.show()
'''
#### FOUND SOLUTIONS ####
sol_array = [[0.12502083,  0.19804898, 12.91606125,  1.76357623], [ 0.37944578,  0.37171809, 13.86560915,  1.98230049]]
init_array = [[0.01,0.01,0.01,0.01], [0.1,0.1,0.1,0.1], [0.3,0.3,0.3,0.3], [0.5,0.5,0.5,0.5], [0.6,0.6,0.6,0.6]]
for i,p in enumerate(sol_array):
    for ind,j in enumerate(init_array):
        time, loss_ar, loss_av, CP_array, CV_array, PA_array, PV_array = rollout3(testObject, j, num_time_steps,p)
        plt.rcParams.update({'font.size': 6})
        plt.subplot(2,5,ind+1+(i*5))
        plt.plot(time, loss_ar)
        p_round = [round(val,2) for val in p]
        plt.xlabel('Time')
        plt.ylabel('Loss')
        plt.title(f"initial = {j}")
plt.show()

for i,p in enumerate(sol_array):
    for ind,j in enumerate(init_array):
        time_array, loss_array, loss_av, CP_array, CV_array, PA_array, PV_array = rollout3(testObject,j,num_time_steps, p)
        plt.rcParams.update({'font.size': 6})

        plt.subplot(2,5,ind+1+(i*5))
        plt.plot(time_array, CP_array, label='CP')
        plt.plot(time_array, CV_array, label='CV')
        plt.plot(time_array, PA_array, label='PA')
        plt.plot(time_array, PV_array, label='PV')
        plt.legend()
        p_round = [round(val,2) for val in p]
        plt.xlabel('Time')
        plt.ylabel('State Variables')
        plt.title(f"initial = {j}")
plt.show()
