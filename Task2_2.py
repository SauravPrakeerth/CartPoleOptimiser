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
'''
trainObject = CartPole()

Xdata, Ydata_CL, Ydata_CV, Ydata_PA, Ydata_PV = [], [], [], [], []

N = 10000

for i in range(N):
    X = [random.uniform(-5,5),random.uniform(-10,10),random.uniform(-np.pi,np.pi),random.uniform(-15,15),random.uniform(-5,5)] # intial state
    trainObject.setState(X[:4])
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
'''
'''
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

#### NON-LINEAR MODEL ####

## Model Training ## 

def alphaGenerator(N,M,lambd):

    N = int(N)
    M = int(M)
    trainObject = CartPole()

    Xdata, Xdata_CL, Xdata_CV, Xdata_PA, Xdata_PV, Xdata_F, Ydata_CL, Ydata_CV, Ydata_PA, Ydata_PV = [], [], [], [], [], [], [], [], [], []

    for i in range(N):
        X = [random.uniform(-5,5),random.uniform(-10,10),random.uniform(-np.pi,np.pi),random.uniform(-15,15), random.uniform(-5,5)] # intial state
        trainObject.setState(X[:4])
        trainObject.performAction(X[4])
        Y = np.array(trainObject.getState()) - np.array(X[:4])
        Xdata.append(X)
        Xdata_CL.append(X[0])
        Xdata_CV.append(X[1])
        Xdata_PA.append(X[2])
        Xdata_PV.append(X[3])
        Xdata_F.append(X[4])
        Ydata_CL.append(Y[0])
        Ydata_CV.append(Y[1])
        Ydata_PA.append(Y[2])
        Ydata_PV.append(Y[3])
        trainObject.reset()

    len_param = [np.std(Xdata_CL), np.std(Xdata_CV), np.std(Xdata_PA), np.std(Xdata_PV), np.std(Xdata_F)]
    Xdata = np.array(Xdata)

    # selecting basis functions
    N = len(Xdata_CL)
    basis_ind = np.random.choice(N,M,replace=False)
    basis = np.array([Xdata[i] for i in basis_ind])
    XdataT = [Xdata_CL,Xdata_CV,Xdata_PA,Xdata_PV, Xdata_F]
    YdataT = [Ydata_CL,Ydata_CV,Ydata_PA,Ydata_PV]

    KNM = kernel_mat(Xdata, basis, len_param)
    KMN = np.transpose(KNM)
    KMM = kernel_mat(basis,basis, len_param)
    a = (np.matmul(KMN,KNM)+lambd*KMM)
    b0 = (np.matmul(KMN, np.array(Ydata_CL)))
    b1 = (np.matmul(KMN, np.array(Ydata_CV)))
    b2 = (np.matmul(KMN, np.array(Ydata_PA)))
    b3 = (np.matmul(KMN, np.array(Ydata_PV)))
    alpha0, res, rank, S = np.linalg.lstsq(a,b0,rcond=None)
    alpha1, res, rank, S = np.linalg.lstsq(a,b1,rcond=None)
    alpha2, res, rank, S = np.linalg.lstsq(a,b2,rcond=None)
    alpha3, res, rank, S = np.linalg.lstsq(a,b3,rcond=None)
    alpha_vec = [alpha0,alpha1,alpha2,alpha3]

    return alpha_vec, basis, len_param, Xdata, Ydata_CL, Ydata_CV, Ydata_PA, Ydata_PV

def kernel_func(x, basis, len_param):
    e_sum = 0
    for i in range(len(x)):
        if i == 2:
            e_sum += ((np.sin((x[i]-basis[i])/2) )**2)/(2*(len_param[i]**2))
        else:
            e_sum += ((x[i] - basis[i])**2)/(2*(len_param[i]**2))
    tot = np.exp(-e_sum)
    return tot


def kernel_mat(X, basis, len_param):
    N = X.shape[0]
    kernel_matrix = np.zeros([N,len(basis)])
    for i,row in enumerate(kernel_matrix):
        for j in range(len(row)):
            kernel_matrix[i][j] = kernel_func(X[i], basis[j], len_param)
    return kernel_matrix

    #### PREDICTION ####
def nonlinearPred(X, alpha_vec, basis, len_param):
    K = kernel_mat(X, basis, len_param)
    M = K.shape[0]
    Y = np.zeros([4,M])
    for m in range(M):
        for j in range(4):
            Yj = np.matmul(K[m],alpha_vec[j])
            Y[j][m] = Yj
    return Y

## PLOTTING YPRED VS Y ##

alpha, basis, len_param, Xdata, Ydata_CL, Ydata_CV, Ydata_PA, Ydata_PV = alphaGenerator(10000,5000,(10**-4))
Ypred_CL, Ypred_CV, Ypred_PA, Ypred_PV = nonlinearPred(Xdata,alpha,basis,len_param)

plt.subplot(1, 4, 1) 
plt.scatter(Ydata_CL, Ypred_CL)
plt.title("Cart Location")
plt.xlabel('Y')
plt.ylabel('Predicted Y')

plt.subplot(1, 4, 2) 
plt.scatter(Ydata_CV, Ypred_CV)
plt.title("Cart Velocity")
plt.xlabel('Y')
plt.ylabel('Predicted Y')

plt.subplot(1, 4, 3) 
plt.scatter(Ydata_PA, Ypred_PA)
plt.title("Pole Angle")
plt.xlabel('Y')
plt.ylabel('Predicted Y')

plt.subplot(1, 4, 4)
plt.scatter(Ydata_PV, Ypred_PV)
plt.title("Pole Velocity")
plt.xlabel('Y')
plt.ylabel('Predicted Y')

plt.show()

'''
## PLOTTING ROLLOUT ##

trainedObject = CartPole()
# initial_state = [0,0,0,6,-3] ## CASE 1
# initial_state = [0,0,3,0,-4]  ## CASE 2
## initial_state = [0,0,np.pi,15,1]  ## CASE 4
initial_state = [2,-2,0,-2,2]  ## CASE 5
num_time_steps = 25
trainedObject.setState(initial_state)
state_array = [initial_state]
alpha, basis, len_param, Xdata, Ydata_CL, Ydata_CV, Ydata_PA, Ydata_PV = alphaGenerator(1000,500,(10**-4))
curr_state = np.array(initial_state)
for t in range(num_time_steps-1):
    curr_change = nonlinearPred(np.array([curr_state]), alpha, basis, len_param)
    curr_change = [i[0] for i in curr_change]
    next_state = np.array(curr_state[:4])+np.array(curr_change)
    next_state[2] = remap_angle(next_state[2])
    state_array.append(next_state)
    next_state = list(next_state)
    next_state.append(curr_state[4])
    curr_state = next_state

testObject = CartPole()
test_CL, test_CV, test_PA, test_PV, time_array = rollout(testObject,initial_state,num_time_steps)


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
alpha, basis, len_param, Xdata, Ydata_CL, Ydata_CV, Ydata_PA, Ydata_PV = alphaGenerator(1000,500,(10**-4))

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
        curr_change = nonlinearPred(np.array([state]), np.array(alpha), basis, len_param)
        y_pred = [i[0] for i in curr_change][sy]
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

alpha, basis, len_param, Xdata, Ydata_CL, Ydata_CV, Ydata_PA, Ydata_PV = alphaGenerator(1000,500,10**(-3))

plotObject = CartPole()

def contour(sx,sy,sz):
    X, Y = np.meshgrid(Xarray[sx], Xarray[sy])
    x, y = np.ravel(X), np.ravel(Y)
    z = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        state = np.array([initial_states])
        state[0][sx], state[0][sy] = x[i], y[i]
        Ypred = nonlinearPred(state, np.array(alpha), basis, len_param)
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

