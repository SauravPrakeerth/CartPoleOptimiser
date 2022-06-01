from cProfile import label
from cgi import test
from pyexpat import model
from pyexpat.errors import XML_ERROR_NOT_STANDALONE
from random import choice
from re import I
from Task1_1 import rollout

from cv2 import INTERSECT_PARTIAL
from CartPole import CartPole, remap_angle
import random
import numpy
import autograd.numpy as np
from matplotlib.pyplot import ion, draw, Rectangle, Line2D
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from math import sqrt

def alphaGenerator(N,M,lambd):

    N = int(N)
    M = int(M)
    trainObject = CartPole()

    Xdata, Xdata_CL, Xdata_CV, Xdata_PA, Xdata_PV, Ydata_CL, Ydata_CV, Ydata_PA, Ydata_PV = [], [], [], [], [], [], [], [], []

    for i in range(N):
        X = [random.uniform(-5,5),random.uniform(-10,10),random.uniform(-np.pi,np.pi),random.uniform(-15,15)] # intial state
        trainObject.setState(X)
        trainObject.performAction()
        Y = np.array(trainObject.getState()) - np.array(X)
        Xdata.append(X)
        Xdata_CL.append(X[0])
        Xdata_CV.append(X[1])
        Xdata_PA.append(X[2])
        Xdata_PV.append(X[3])
        Ydata_CL.append(Y[0])
        Ydata_CV.append(Y[1])
        Ydata_PA.append(Y[2])
        Ydata_PV.append(Y[3])
        trainObject.reset()

    len_param = [np.std(Xdata_CL), np.std(Xdata_CV), np.std(Xdata_PA), np.std(Xdata_PV)]
    Xdata = np.array(Xdata)

    # selecting basis functions
    N = len(Xdata_CL)
    basis_ind = np.random.choice(N,M,replace=False)
    basis = np.array([Xdata[i] for i in basis_ind])
    XdataT = [Xdata_CL,Xdata_CV,Xdata_PA,Xdata_PV]
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

'''
#### M ERROR ####

er_mat = []
M_test = np.linspace(10,1000,20)
for M in M_test:
    alpha, basis, len_param, Xdata, Ydata_CL, Ydata_CV, Ydata_PA, Ydata_PV = alphaGenerator(1000,M,10**(-3))
    Ypred = nonlinearPred(Xdata, np.array(alpha), basis, len_param)
    er_list = [mean_squared_error(Ydata_CL, Ypred[0]), mean_squared_error(Ydata_CV, Ypred[1]), mean_squared_error(Ydata_PA, Ypred[2]), mean_squared_error(Ydata_PV, Ypred[3])]
    er_mat.append(er_list)

MSE_CL = [er_mat[i][0] for i in range(len(er_mat))]
MSE_CV = [er_mat[i][1] for i in range(len(er_mat))]
MSE_PA = [er_mat[i][2] for i in range(len(er_mat))]
MSE_PV = [er_mat[i][3] for i in range(len(er_mat))]

plt.subplot(1,4,1)
plt.plot(M_test ,MSE_CL)
plt.title("Cart Location")
plt.xlabel('M')
plt.ylabel('MSE')

plt.subplot(1,4,2)
plt.plot(M_test ,MSE_CV)
plt.title("Cart Location")
plt.xlabel('M')
plt.ylabel('MSE')

plt.subplot(1,4,3)
plt.plot(M_test ,MSE_PA)
plt.title("Cart Location")
plt.xlabel('M')
plt.ylabel('MSE')

plt.subplot(1,4,4)
plt.plot(M_test ,MSE_PV)
plt.title("Cart Location")
plt.xlabel('M')
plt.ylabel('MSE')
plt.show()



#### LAMBDA ERROR ####

er_mat = []
L_test = np.linspace(10**(-6),10**(-1))
for L in L_test:
    alpha, basis, len_param, Xdata, Ydata_CL, Ydata_CV, Ydata_PA, Ydata_PV = alphaGenerator(500,100,L)
    Ypred = nonlinearPred(Xdata, np.array(alpha), basis, len_param)
    er_list = [mean_squared_error(Ydata_CL, Ypred[0]), mean_squared_error(Ydata_CV, Ypred[1]), mean_squared_error(Ydata_PA, Ypred[2]), mean_squared_error(Ydata_PV, Ypred[3])]
    er_mat.append(er_list)

MSE_CL = [er_mat[i][0] for i in range(len(er_mat))]
MSE_CV = [er_mat[i][1] for i in range(len(er_mat))]
MSE_PA = [er_mat[i][2] for i in range(len(er_mat))]
MSE_PV = [er_mat[i][3] for i in range(len(er_mat))]

plt.subplot(1,4,1)
plt.plot(L_test ,MSE_CL)
plt.title("Cart Location")
plt.xlabel('Lambda')
plt.ylabel('MSE')

plt.subplot(1,4,2)
plt.plot(L_test ,MSE_CV)
plt.title("Cart Location")
plt.xlabel('Lambda')
plt.ylabel('MSE')

plt.subplot(1,4,3)
plt.plot(L_test ,MSE_PA)
plt.title("Cart Location")
plt.xlabel('Lambda')
plt.ylabel('MSE')

plt.subplot(1,4,4)
plt.plot(L_test ,MSE_PV)
plt.title("Cart Location")
plt.xlabel('Lambda')
plt.ylabel('MSE')
plt.show()


#### N ERROR ####

er_mat = []
N_test = np.linspace(100,1000,10)
B_test = np.linspace(10,100,10)
for i in range(len(N_test)):
    alpha, basis, len_param, Xdata, Ydata_CL, Ydata_CV, Ydata_PA, Ydata_PV = alphaGenerator(N_test[i],B_test[i],(10**-4))
    Ypred = nonlinearPred(Xdata, np.array(alpha), basis, len_param)
    er_list = [mean_squared_error(Ydata_CL, Ypred[0]), mean_squared_error(Ydata_CV, Ypred[1]), mean_squared_error(Ydata_PA, Ypred[2]), mean_squared_error(Ydata_PV, Ypred[3])]
    er_mat.append(er_list)

MSE_CL = [er_mat[i][0] for i in range(len(er_mat))]
MSE_CV = [er_mat[i][1] for i in range(len(er_mat))]
MSE_PA = [er_mat[i][2] for i in range(len(er_mat))]
MSE_PV = [er_mat[i][3] for i in range(len(er_mat))]

plt.subplot(1,4,1)
plt.plot(N_test ,MSE_CL)
plt.title("Cart Location")
plt.xlabel('N')
plt.ylabel('MSE')

plt.subplot(1,4,2)
plt.plot(N_test ,MSE_CV)
plt.title("Cart Location")
plt.xlabel('N')
plt.ylabel('MSE')

plt.subplot(1,4,3)
plt.plot(N_test ,MSE_PA)
plt.title("Cart Location")
plt.xlabel('N')
plt.ylabel('MSE')

plt.subplot(1,4,4)
plt.plot(N_test ,MSE_PV)
plt.title("Cart Location")
plt.xlabel('N')
plt.ylabel('MSE')
plt.show()

'''
'''
#### Ypred against Y ####
alpha, basis, len_param, Xdata, Ydata_CL, Ydata_CV, Ydata_PA, Ydata_PV = alphaGenerator(1000,500,(10**-4))
Ypred_CL, Ypred_CV, Ypred_PA, Ypred_PV = nonlinearPred(Xdata,alpha,basis,len_param)

plt.subplot(2, 2, 1) 
plt.scatter(Ydata_CL, Ypred_CL)
plt.title("Cart Location")
plt.xlabel('Y')
plt.ylabel('Predicted Y')

plt.subplot(2, 2, 2) 
plt.scatter(Ydata_CV, Ypred_CV)
plt.title("Cart Velocity")
plt.xlabel('Y')
plt.ylabel('Predicted Y')

plt.subplot(2, 2, 3) 
plt.scatter(Ydata_PA, Ypred_PA)
plt.title("Pole Angle")
plt.xlabel('Y')
plt.ylabel('Predicted Y')

plt.subplot(2, 2, 4)
plt.scatter(Ydata_PV, Ypred_PV)
plt.title("Pole Velocity")
plt.xlabel('Y')
plt.ylabel('Predicted Y')

plt.show()
'''
#### ROLLOOUT ####

trainedObject = CartPole()
# initial_state = [0,0,0,6] ## CASE 1
initial_state = [0,0,np.pi,5]  ## CASE 2
# initial_state = [2,-2,0,-2]  ## CASE 3
num_time_steps = 25
trainedObject.setState(initial_state)
state_array = [initial_state]
alpha, basis, len_param, Xdata, Ydata_CL, Ydata_CV, Ydata_PA, Ydata_PV = alphaGenerator(10000,5000,(10**-4))
curr_state = np.array(initial_state)
for t in range(num_time_steps-1):
    curr_change = nonlinearPred(np.array([curr_state]), np.array(alpha), basis, len_param)
    curr_change = [i[0] for i in curr_change]
    next_state = curr_state+curr_change
    next_state[2] = remap_angle(next_state[2])
    state_array.append(next_state)
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
# RMSE ROLLOUT # 

CL_MSE_time = [np.abs(x-y) for x,y in zip(test_CL, [state_array[i][0] for i in range(np.shape(state_array)[0])])]
CV_MSE_time = [np.abs(x-y) for x,y in zip(test_CV, [state_array[i][1] for i in range(np.shape(state_array)[0])])]
PA_MSE_time = [np.abs(x-y) for x,y in zip(test_PA, [state_array[i][2] for i in range(np.shape(state_array)[0])])]
PV_MSE_time = [np.abs(x-y) for x,y in zip(test_PV, [state_array[i][3] for i in range(np.shape(state_array)[0])])]

plt.subplot(1, 4, 1)
plt.plot(time_array, CL_MSE_time)
plt.title("Cart Location")
plt.xlabel('Time')
plt.ylabel('Absolute error')

plt.subplot(1, 4, 2)
plt.plot(time_array, CV_MSE_time)
plt.title("Cart Velocity")
plt.xlabel('Time')
plt.ylabel('Absolute error')

plt.subplot(1, 4, 3)
plt.plot(time_array, PA_MSE_time)
plt.title("Pole Angle")
plt.xlabel('Time')
plt.ylabel('Absolute error')

plt.subplot(1, 4, 4)
plt.plot(time_array, PV_MSE_time)
plt.title("Pole Velocity")
plt.xlabel('Time')
plt.ylabel('Absolute error')

plt.show()
'''
#### CONTOUR PLOTS ####

'''
initial_states = [random.uniform(-5,5), random.uniform(-10,10), random.uniform(-np.pi,np.pi), random.uniform(-15,15)]
Xarray = [[cart_loc for cart_loc in range(-5,5)],[cart_vel for cart_vel in range(-10,10)],[pole_ang for pole_ang in numpy.linspace(-np.pi,np.pi,30)], [pole_vel for pole_vel in range(-15,15)]]
labels = ['CL', 'CV', 'PA', 'PV']

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

test_xyz = [(0,1,0),(0,1,1),(0,1,2),(0,1,3), (0,2,0),(0,2,1),(0,2,2),(0,2,3), (0,3,0),(0,3,1),(0,3,2),(0,3,3), (1,2,0),(1,2,1),(1,2,2),(1,2,3), (1,3,0),(1,3,1),(1,3,2),(1,3,3), (2,3,0),(2,3,1),(2,3,2),(2,3,3)]

contourPlot(test_xyz, (4,6))
'''