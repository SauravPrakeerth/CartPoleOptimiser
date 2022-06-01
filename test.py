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

print(np.random.normal(0,1,1)[0])