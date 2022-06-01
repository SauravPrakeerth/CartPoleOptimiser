from CartPole import CartPole, remap_angle
#import numpy as np
import autograd.numpy as np
from matplotlib.pyplot import ion, draw, Rectangle, Line2D
import matplotlib.pyplot as plt

def rollout(Object,initial_state,num_time_steps):

    Object.setState(initial_state)

    time_array = []
    CP_array = []
    CV_array = []
    PA_array = []
    PV_array = []
    X = []

    for i in range(num_time_steps):
        Object.performAction()
        time_array.append(i)
        X = Object.getState()
        CP_array.append(X[0])
        CV_array.append(X[1])
        PA_array.append(remap_angle(X[2]))
        PV_array.append(X[3])
    
    return CP_array, CV_array, PA_array, PV_array, time_array

# TESTING #

'''
# State variable against time plots
plt.subplot(1,4,1)
plt.plot(time_array,CP_array)
plt.xlabel("time")
plt.ylabel("x")
plt.title("Cart Location")

plt.subplot(1,4,2)
plt.plot(time_array,CV_array)
plt.xlabel("time")
plt.ylabel("ẋ")
plt.title("Cart Velocity")

plt.subplot(1,4,3)
plt.plot(time_array,PA_array)
plt.xlabel("Time")
plt.ylabel("θ")
plt.title("Pole Angle")

plt.subplot(1,4,4)
plt.plot(time_array,PV_array)
plt.xlabel("Time")
plt.ylabel("θ˙")
plt.title("Pole Velocity")
plt.show()
'''
testObject = CartPole()
CP_array, CV_array, PA_array, PV_array, time_array = rollout(testObject,[2,-2,0,-2],6)

# initial_state = [0,0,0,6] ## CASE 1
# initial_state = [0,0,3,0]  ## CASE 2
# initial_state = [0,0,np.pi,15]  ## CASE 4
# initial_state = [2,-2,0,-2]  ## CASE 5
