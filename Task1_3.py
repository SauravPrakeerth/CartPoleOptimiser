from lib2to3.pgen2.literals import test
from random import choice
from CartPole import CartPole
import random
import numpy
import autograd.numpy as np
from matplotlib.pyplot import ion, draw, Rectangle, Line2D
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

testObject = CartPole(visual=False)

Xdata, Ydata_CL, Ydata_CV, Ydata_PA, Ydata_PV = [], [], [], [], []

for i in range(500):
    X = [random.uniform(-5,5),random.uniform(-10,10),random.uniform(-np.pi,np.pi),random.uniform(-15,15)] # intial state
    testObject.setState(X)
    testObject.performAction()
    Y = np.array(testObject.getState()) - np.array(X)
    Xdata.append(X)
    Ydata_CL.append(Y[0])
    Ydata_CV.append(Y[1])
    Ydata_PA.append(Y[2])
    Ydata_PV.append(Y[3])
    testObject.reset()

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

# Plotting 
plt.subplot(2, 2, 1) 
plt.scatter(Ydata_CL, model_CL.predict(Xdata))
plt.title("Cart Location")
plt.xlabel('Y')
plt.ylabel('Predicted Y')

plt.subplot(2, 2, 2) 
plt.scatter(Ydata_CV, model_CV.predict(Xdata))
plt.title("Cart Velocity")
plt.xlabel('Y')
plt.ylabel('Predicted Y')

plt.subplot(2, 2, 3) 
plt.scatter(Ydata_PA, model_PA.predict(Xdata))
plt.title("Pole Angle")
plt.xlabel('Y')
plt.ylabel('Predicted Y')

plt.subplot(2, 2, 4)
plt.scatter(Ydata_PV, model_PV.predict(Xdata))
plt.title("Pole Velocity")
plt.xlabel('Y')
plt.ylabel('Predicted Y')

plt.show()


#### SCANS ####

testObject = CartPole(visual=False)

initial_CL = random.uniform(-5,5)
initial_CV = random.uniform(-10,10)
initial_PA = random.uniform(-np.pi,np.pi)
initial_PV = random.uniform(-15,15)

CL_Xarray, CV_Xarray, PA_Xarray, PV_Xarray, CL_Yarray, CV_Yarray, PA_Yarray, PV_Yarray, Xdata = [], [], [], [], [], [], [], [], []
# Varying pole velocity
for pole_vel in range(-15,15):
    X = [initial_CL,initial_CV,initial_PA,pole_vel] # intial state
    PV_Xarray.append(X[3]) # pole velocity X
    Xdata = list(Xdata)
    Xdata.append(X)
    Xdata = (np.array(Xdata)).reshape(-1,4) # reshaping

    # Real data
    testObject.setState(X)
    testObject.performAction()
    Y = np.array(testObject.getState()) - np.array(X)
    PV_Yarray.append(Y[3])
    testObject.reset()

plt.subplot (2,2,4)
plt.plot(PV_Xarray, PV_Yarray, label = "real next change")
plt.plot(PV_Xarray, model_PV.predict(Xdata), label = "predicted next change")
plt.legend()
plt.title('Scanning Pole Velocity')
plt.xlabel('Pole Velocity X')
plt.ylabel('Pole Velocity Y')


# Varying pole angle
Xdata = []
for pole_ang in numpy.linspace(-np.pi,np.pi,30):
    X = [initial_CL,initial_CV,pole_ang,initial_PV] # initial state
    PA_Xarray.append(X[2]) # pole angle X
    Xdata = list(Xdata)
    Xdata.append(X)
    Xdata = (np.array(Xdata)).reshape(-1,4) # reshaping

    # Real data
    testObject.setState(X)
    testObject.performAction()
    Y = np.array(testObject.getState()) - np.array(X)
    PA_Yarray.append(Y[2])
    testObject.reset()

plt.subplot(2,2,3)
plt.plot(PA_Xarray, PA_Yarray, label = "real next change")
plt.plot(PA_Xarray, model_PA.predict(Xdata), label = "predicted next change")
plt.legend()
plt.title('Scanning Pole Angle')
plt.xlabel('Pole Angle X')
plt.ylabel('Pole Angle Y')


# Varying cart velocity
Xdata = []
for cart_vel in range(-10,10):
    X = [initial_CL,cart_vel,initial_PA,initial_PV] # intial state
    CV_Xarray.append(X[1]) # cart velocity X
    Xdata = list(Xdata)
    Xdata.append(X)
    Xdata = (np.array(Xdata)).reshape(-1,4) # reshaping

    # Real data
    testObject.setState(X)
    testObject.performAction()
    Y = np.array(testObject.getState()) - np.array(X)
    CV_Yarray.append(Y[1])
    testObject.reset()

plt.subplot(2,2,2)
plt.plot(CV_Xarray, CV_Yarray, label = "real next change")
plt.plot(CV_Xarray, model_CV.predict(Xdata), label = "predicted next change")
plt.legend()
plt.title('Scanning Cart Velocity')
plt.xlabel('Cart Velocity X')
plt.ylabel('Cart Velocity Y')

# Varying cart position - Has no effect on the next step
Xdata = []
for cart_loc in range(-5,5):
    X = [cart_loc,initial_CV,initial_PA,initial_PV] # intial state
    CL_Xarray.append(X[0]) # cart location X
    Xdata = list(Xdata)
    Xdata.append(X)
    Xdata = (np.array(Xdata)).reshape(-1,4) # reshaping

    # Real data
    testObject.setState(X)
    testObject.performAction()
    Y = np.array(testObject.getState()) - np.array(X)
    CL_Yarray.append(Y[0])
    testObject.reset()

plt.subplot(2,2,1)
plt.plot(CL_Xarray, CL_Yarray, label = "real next change")
plt.plot(CL_Xarray, model_CL.predict(Xdata), label = "predicted next change")
plt.legend()
plt.title('Scanning Cart Location')
plt.xlabel('Cart Location X')
plt.ylabel('Cart Location Y')
plt.show()

