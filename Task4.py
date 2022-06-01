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

def weightGenerator(N,M,lambd):

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