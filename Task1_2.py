from random import choice
from CartPole import CartPole
import random
import numpy
import autograd.numpy as np
from matplotlib.pyplot import ion, draw, Rectangle, Line2D
import matplotlib.pyplot as plt
import plotly.graph_objects as go


testObject = CartPole(visual=False)

initial_CL = random.uniform(-5,5)
initial_CV = random.uniform(-10,10)
initial_PA = random.uniform(-np.pi,np.pi)
initial_PV = random.uniform(-15,15)

CL_Xarray, CV_Xarray, PA_Xarray, PV_Xarray, CL_Yarray, CV_Yarray, PA_Yarray, PV_Yarray = [], [], [], [], [], [], [], []

#### Defining Y as next state ####

# SCANNING CART LOCATION
for cart_loc in range(-5,5):
    X = [cart_loc,initial_CV,initial_PA,initial_PV] # intial state
    CL_Xarray.append(X[0]) 
    testObject.setState(X)
    testObject.performAction()
    Y = testObject.getState()
    CL_Yarray.append(Y[0])
    testObject.reset()

plt.subplot(1,4,1)
plt.plot(CL_Xarray, CL_Yarray)
plt.title('Scanning Cart Location')
plt.xlabel('Cart Location X')
plt.ylabel('Cart Location Y')

# SCANNING CART VELOCITY
for cart_vel in range(-10,10):
    X = [initial_CL,cart_vel,initial_PA,initial_PV] # intial state
    CV_Xarray.append(X[1]) 
    testObject.setState(X)
    testObject.performAction()
    Y = testObject.getState()
    CV_Yarray.append(Y[1])
    testObject.reset()

plt.subplot(1,4,2)
plt.plot(CV_Xarray, CV_Yarray)
plt.title('Scanning Cart Velocity')
plt.xlabel('Cart Velocity X')
plt.ylabel('Cart Velocity Y')

# SCANNING POLE ANGLE
for pole_ang in np.linspace(-np.pi,np.pi,10):
    X = [initial_CL,initial_CV,pole_ang,initial_PV] # intial state
    PA_Xarray.append(X[2]) 
    testObject.setState(X)
    testObject.performAction()
    Y = testObject.getState()
    PA_Yarray.append(Y[2]) 
    testObject.reset()

plt.subplot(1,4,3)
plt.plot(PA_Xarray, PA_Yarray)
plt.title('Scanning Pole Angle')
plt.xlabel('Pole Angle X')
plt.ylabel('Pole Angle Y')


# SCANNING POLE VELOCITY
for pole_vel in range(-15,15):
    X = [initial_CL,initial_CV,initial_PA,pole_vel] # intial state
    PV_Xarray.append(X[3]) # pole velocity X
    testObject.setState(X)
    testObject.performAction()
    Y = testObject.getState()
    PV_Yarray.append(Y[3]) # pole velocity Y
    testObject.reset()

plt.subplot(1,4,4)
plt.plot(PV_Xarray, PV_Yarray)
plt.title('Scanning Pole Velocity')
plt.xlabel('Pole Velocity X')
plt.ylabel('Pole Velocity Y')
plt.show()

#### Defining Y as change ####

# Varying pole velocity
for pole_vel in range(-15,15):
    X = [initial_CL,initial_CV,initial_PA,pole_vel] # intial state
    CL_Xarray.append(X[0]) # cart location X
    CV_Xarray.append(X[1]) # cart velocity X
    PA_Xarray.append(X[2]) # pole angle X
    PV_Xarray.append(X[3]) # pole velocity X
    testObject.setState(X)
    testObject.performAction()
    Y = np.array(testObject.getState()) - np.array(X)
    CL_Yarray.append(Y[0]) # cart location Y
    CV_Yarray.append(Y[1]) # cart velocity Y
    PA_Yarray.append(Y[2]) # pole angle Y
    PV_Yarray.append(Y[3]) # pole velocity Y
    testObject.reset()

plt.subplot (2,2,4)
plt.plot(PV_Xarray, PV_Yarray)
plt.title('Scanning Pole Velocity')
plt.xlabel('Pole Velocity X')
plt.ylabel('Pole Velocity Y')


# Varying pole angle
CL_Xarray, CV_Xarray, PA_Xarray, PV_Xarray, CL_Yarray, CV_Yarray, PA_Yarray, PV_Yarray = [], [], [], [], [], [], [], []
for pole_ang in numpy.linspace(-np.pi,np.pi,30):
    X = [initial_CL,initial_CV,pole_ang,initial_PV] # intial state
    CL_Xarray.append(X[0]) # cart location X
    CV_Xarray.append(X[1]) # cart velocity X
    PA_Xarray.append(X[2]) # pole angle X
    PV_Xarray.append(X[3]) # pole velocity X
    testObject.setState(X)
    testObject.performAction()
    Y = np.array(testObject.getState()) - np.array(X)
    CL_Yarray.append(Y[0]) # cart location Y
    CV_Yarray.append(Y[1]) # cart velocity Y
    PA_Yarray.append(Y[2]) # pole angle Y
    PV_Yarray.append(Y[3]) # pole velocity Y
    testObject.reset()

plt.subplot(2,2,3)
plt.plot(PA_Xarray, PA_Yarray)
plt.title('Scanning Pole Angle')
plt.xlabel('Pole Angle X')
plt.ylabel('Pole Angle Y')


# Varying cart velocity
CL_Xarray, CV_Xarray, PA_Xarray, PV_Xarray, CL_Yarray, CV_Yarray, PA_Yarray, PV_Yarray = [], [], [], [], [], [], [], []
for cart_vel in range(-10,10):
    X = [initial_CL,cart_vel,initial_PA,initial_PV] # intial state
    CL_Xarray.append(X[0]) # cart location X
    CV_Xarray.append(X[1]) # cart velocity X
    PA_Xarray.append(X[2]) # pole angle X
    PV_Xarray.append(X[3]) # pole velocity X
    testObject.setState(X)
    testObject.performAction()
    Y = np.array(testObject.getState()) - np.array(X)
    CL_Yarray.append(Y[0]) # cart location Y
    CV_Yarray.append(Y[1]) # cart velocity Y
    PA_Yarray.append(Y[2]) # pole angle Y
    PV_Yarray.append(Y[3]) # pole velocity Y
    testObject.reset()

plt.subplot(2,2,2)
plt.plot(CV_Xarray, CV_Yarray)
plt.title('Scanning Cart Velocity')
plt.xlabel('Cart Velocity X')
plt.ylabel('Cart Velocity Y')


# Varying cart position - Has no effect on the next step
CL_Xarray, CV_Xarray, PA_Xarray, PV_Xarray, CL_Yarray, CV_Yarray, PA_Yarray, PV_Yarray = [], [], [], [], [], [], [], []
for cart_loc in range(-5,5):
    X = [cart_loc,initial_CL,initial_PA,initial_PV] # intial state
    CL_Xarray.append(X[0]) # cart location X
    CV_Xarray.append(X[1]) # cart velocity X
    PA_Xarray.append(X[2]) # pole angle X
    PV_Xarray.append(X[3]) # pole velocity X
    testObject.setState(X)
    testObject.performAction()
    Y = np.array(testObject.getState()) - np.array(X)
    CL_Yarray.append(Y[0]) # cart location Y
    CV_Yarray.append(Y[1]) # cart )*velocity Y
    PA_Yarray.append(Y[2]) # pole angle Y
    PV_Yarray.append(Y[3]) # pole velocity Y
    testObject.reset()

plt.subplot(2,2,1)
plt.plot(CL_Xarray, CL_Yarray)
plt.title('Scanning Cart Location')
plt.xlabel('Cart Location X')
plt.ylabel('Cart Location Y')
plt.show()

#### CONTOUR PLOTS ####

plotObject = CartPole()

CL_Xarray = [cart_loc for cart_loc in range(-5,5)]
CV_Xarray = [cart_vel for cart_vel in range(-10,10)]
PA_Xarray = [pole_ang for pole_ang in numpy.linspace(-np.pi,np.pi,30)]
PV_Xarray = [pole_vel for pole_vel in range(-15,15)]

# CLx and CVx
x = CL_Xarray
y = CV_Xarray

X, Y = np.meshgrid(CL_Xarray, CV_Xarray)
x, y = np.ravel(X), np.ravel(Y)
z = np.zeros(x.shape[0])
for i in range(x.shape[0]):
    plotObject.setState([x[i],y[i],initial_PA, initial_PV])
    plotObject.performAction()
    z[i] = plotObject.getState()[0]
plt.rcParams.update({'font.size': 5})
plt.subplot(4,6,1)
plt.tricontourf(x,y,z,levels=12)
plt.title("CL = f(CL, CV)")

z = np.zeros(x.shape[0])
for i in range(x.shape[0]):
    plotObject.setState([x[i],y[i],initial_PA, initial_PV])
    plotObject.performAction()
    z[i] = plotObject.getState()[1]
plt.subplot(4,6,7)
plt.tricontourf(x,y,z,levels=12)
plt.title("CV = f(CL, CV)")

z = np.zeros(x.shape[0])
for i in range(x.shape[0]):
    plotObject.setState([x[i],y[i],initial_PA, initial_PV])
    plotObject.performAction()
    z[i] = plotObject.getState()[2]
plt.subplot(4,6,13)
plt.tricontourf(x,y,z,levels=12)
plt.title("Pole Angle = f(CL, CV)")

z = np.zeros(x.shape[0])
for i in range(x.shape[0]):
    plotObject.setState([x[i],y[i],initial_PA, initial_PV])
    plotObject.performAction()
    z[i] = plotObject.getState()[3]
plt.subplot(4,6,19)
plt.tricontourf(x,y,z,levels=12)
plt.title("PV = f(CL, CV)")


# CLx and PAx
x = CL_Xarray
y = PA_Xarray
X, Y = np.meshgrid(CL_Xarray, PA_Xarray)
x, y = np.ravel(X), np.ravel(Y)
z = np.zeros(x.shape[0])
for i in range(x.shape[0]):
    plotObject.setState([x[i],initial_CV,y[i],initial_PV])
    plotObject.performAction()
    z[i] = plotObject.getState()[0]
plt.subplot(4,6,2)
plt.tricontourf(x,y,z,levels=12)
plt.title("CL = f(CL, Pole Angle)")

z = np.zeros(x.shape[0])
for i in range(x.shape[0]):
    plotObject.setState([x[i],initial_CV,y[i],initial_PV])
    plotObject.performAction()
    z[i] = plotObject.getState()[1]
plt.subplot(4,6,8)
plt.tricontourf(x,y,z,levels=12)
plt.title("CV = f(CL, Pole Angle)")

z = np.zeros(x.shape[0])
for i in range(x.shape[0]):
    plotObject.setState([x[i],initial_CV,y[i], initial_PV])
    plotObject.performAction()
    z[i] = plotObject.getState()[2]
plt.subplot(4,6,14)
plt.tricontourf(x,y,z,levels=12)
plt.title("Pole Angle = f(CL, Pole Angle)")

z = np.zeros(x.shape[0])
for i in range(x.shape[0]):
    plotObject.setState([x[i],initial_CV,y[i], initial_PV])
    plotObject.performAction()
    z[i] = plotObject.getState()[3]
plt.subplot(4,6,20)
plt.tricontourf(x,y,z,levels=12)
plt.title("PV = f(CL, PA)")


# CLx and PVx
x = CL_Xarray
y = PV_Xarray

X, Y = np.meshgrid(CL_Xarray, PV_Xarray)
x, y = np.ravel(X), np.ravel(Y)
z = np.zeros(x.shape[0])
for i in range(x.shape[0]):
    plotObject.setState([x[i],initial_CV,initial_PA, y[i]])
    plotObject.performAction()
    z[i] = plotObject.getState()[0]
plt.subplot(4,6,3)
plt.tricontourf(x,y,z,levels=12)
plt.title("CL = f(CL, PV)")


z = np.zeros(x.shape[0])
for i in range(x.shape[0]):
    plotObject.setState([x[i],initial_CV,initial_PA, y[i]])
    plotObject.performAction()
    z[i] = plotObject.getState()[1]
plt.subplot(4,6,9)
plt.tricontourf(x,y,z,levels=12)
plt.title("CV = f(CL, PV)")

z = np.zeros(x.shape[0])

for i in range(x.shape[0]):
    plotObject.setState([x[i],initial_CV,initial_PA, y[i]])
    plotObject.performAction()
    z[i] = plotObject.getState()[2]
plt.subplot(4,6,15)
plt.tricontourf(x,y,z,levels=12)
plt.title("PA = f(CL, PV)")

z = np.zeros(x.shape[0])

for i in range(x.shape[0]):
    plotObject.setState([x[i],initial_CV,initial_PA, y[i]])
    plotObject.performAction()
    z[i] = plotObject.getState()[3]
plt.subplot(4,6,21)
plt.tricontourf(x,y,z,levels=12)
plt.title("PV = f(CL, PV)")


# CVx and PAx
x = CV_Xarray
y = PA_Xarray
X, Y = np.meshgrid(CV_Xarray, PA_Xarray)
x, y = np.ravel(X), np.ravel(Y)
z = np.zeros(x.shape[0])
for i in range(x.shape[0]):
    plotObject.setState([initial_CL,x[i],y[i],initial_PV])
    plotObject.performAction()
    z[i] = plotObject.getState()[0]
plt.subplot(4,6,4)
plt.tricontourf(x,y,z,levels=12)
plt.title("CL = f(CV, PA)")

z = np.zeros(x.shape[0])
for i in range(x.shape[0]):
    plotObject.setState([initial_CL,x[i],y[i],initial_PV])
    plotObject.performAction()
    z[i] = plotObject.getState()[1]
plt.subplot(4,6,10)
plt.tricontourf(x,y,z,levels=12)
plt.title("CV = f(CV, PA)")

z = np.zeros(x.shape[0])
for i in range(x.shape[0]):
    plotObject.setState([initial_CL,x[i],y[i],initial_PV])
    plotObject.performAction()
    z[i] = plotObject.getState()[2]
plt.subplot(4,6,16)
plt.tricontourf(x,y,z,levels=12)
plt.title("PA = f(CV, PA)")

z = np.zeros(x.shape[0])
for i in range(x.shape[0]):
    plotObject.setState([initial_CL,x[i],y[i],initial_PV])
    plotObject.performAction()
    z[i] = plotObject.getState()[3]
plt.subplot(4,6,22)
plt.tricontourf(x,y,z,levels=12)
plt.title("PV = f(CV, PA)")


# CVx and PVx
x = CV_Xarray
y = PV_Xarray
X, Y = np.meshgrid(CV_Xarray, PV_Xarray)
x, y = np.ravel(X), np.ravel(Y)
z = np.zeros(x.shape[0])
for i in range(x.shape[0]):
    plotObject.setState([initial_CL,x[i],initial_PA,y[i]])
    plotObject.performAction()
    z[i] = plotObject.getState()[0]
plt.subplot(4,6,5)
plt.tricontourf(x,y,z,levels=12)
plt.title("CL = f(CV, PV)")

z = np.zeros(x.shape[0])
for i in range(x.shape[0]):
    plotObject.setState([initial_CL,x[i],initial_PA,y[i]])
    plotObject.performAction()
    z[i] = plotObject.getState()[1]
plt.subplot(4,6,11)
plt.tricontourf(x,y,z,levels=12)
plt.title("CV = f(CV, PV)")

z = np.zeros(x.shape[0])
for i in range(x.shape[0]):
    plotObject.setState([initial_CL,x[i],initial_PA,y[i]])
    plotObject.performAction()
    z[i] = plotObject.getState()[2]
plt.subplot(4,6,17)
plt.tricontourf(x,y,z,levels=12)
plt.title("PA = f(CV, PV)")

z = np.zeros(x.shape[0])
for i in range(x.shape[0]):
    plotObject.setState([initial_CL,x[i],initial_PA,y[i]])
    plotObject.performAction()
    z[i] = plotObject.getState()[3]
plt.subplot(4,6,23)
plt.tricontourf(x,y,z,levels=12)
plt.title("PV = f(CV, PV)")


# PAx and PVx
x = PA_Xarray
y = PV_Xarray
X, Y = np.meshgrid(PA_Xarray, PV_Xarray)
x, y = np.ravel(X), np.ravel(Y)
z = np.zeros(x.shape[0])
for i in range(x.shape[0]):
    plotObject.setState([initial_CL,initial_CV,x[i],y[i]])
    plotObject.performAction()
    z[i] = plotObject.getState()[0]
plt.subplot(4,6,6)
plt.tricontourf(x,y,z,levels=12)
plt.title("CL = f(PA, PV)")

z = np.zeros(x.shape[0])
for i in range(x.shape[0]):
    plotObject.setState([initial_CL,initial_CV,x[i],y[i]])
    plotObject.performAction()
    z[i] = plotObject.getState()[1]
plt.subplot(4,6,12)
plt.tricontourf(x,y,z,levels=12)
plt.title("CV = f(PA, PV)")

z = np.zeros(x.shape[0])
for i in range(x.shape[0]):
    plotObject.setState([initial_CL,initial_CV,x[i],y[i]])
    plotObject.performAction()
    z[i] = plotObject.getState()[2]
plt.subplot(4,6,18)
plt.tricontourf(x,y,z,levels=12)
plt.title("PA = f(PA, PV)")

z = np.zeros(x.shape[0])
for i in range(x.shape[0]):
    plotObject.setState([initial_CL,initial_CV,x[i],y[i]])
    plotObject.performAction()
    z[i] = plotObject.getState()[3]
plt.subplot(4,6,24)
plt.tricontourf(x,y,z,levels=12)
plt.title("PV = f(PA, PV)")
plt.show()