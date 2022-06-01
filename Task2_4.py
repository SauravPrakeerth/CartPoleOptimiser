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

#### ROLLOOUT ####

alpha, basis, len_param, Xdata, Ydata_CL, Ydata_CV, Ydata_PA, Ydata_PV = alphaGenerator(1000,500,(10**-4))

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

def objective(p):
    initial_cond = [0,0,0.01,0]
    num_time_steps = 6
    Object = CartPole(visual=False)
    Object.setState(initial_cond)
    X = initial_cond
    loss_array = [loss(X)]

    for i in range(num_time_steps):
        # update X
        X.append(p[0]*X[0] + p[1]*X[1] + p[2]*X[2] + p[3]*X[3])
        X = nonlinearPred(np.array([X]), np.array(alpha), basis, len_param)
        X = [i[0] for i in X]
        loss_array.append(loss(X))
    loss_av = np.mean(loss_array)
    return loss_av

sol_array = []
for i in range(5):
    pt = [random.uniform(-10,10), random.uniform(-10,10), random.uniform(-10,10), random.uniform(-10,10)]
    result = minimize(objective, pt, method='Nelder-Mead')
    solution = result.x
    sol_array.append(solution)
print(sol_array)

initial_cond = np.array([0,0,0.01,0])
num_time_steps = 15 # NON-LINEAR MODEL VALID TILL ABOUT 15 VIMP
testObject = CartPole()

for i,p in enumerate(sol_array):
    time, loss_ar, loss_av, CP_array, CV_array, PA_array, PV_array = rollout3(testObject, initial_cond, num_time_steps,p)
    # plt.rcParams.update({'font.size': 5})
    plt.subplot(1,5,i+1)
    plt.plot(time, loss_ar)
    p_round = [round(val,2) for val in p]
    plt.xlabel('Time')
    plt.ylabel('Loss')
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
    plt.xlabel('Time')
    plt.ylabel('State Variable')
    plt.legend()
    p_round = [round(val,2) for val in p]
    plt.title(f"p = {p_round}")
plt.show()
