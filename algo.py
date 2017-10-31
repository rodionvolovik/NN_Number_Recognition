import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io#Used to load the OCTAVE *.mat files
import scipy.misc#Used to show matrix as an image
import matplotlib.cm as cm#Used to display images in a specific colormap
import random#To pick random images to display
from scipy.special import expit#Vectorized sigmoid function
from scipy import optimize

def h(mytheta, myX):
    return expit(np.dot(myX, mytheta))


def computeCost(mytheta, myX, myy, mylambda = 0.):
    m = myX.shape[0]
    myh = h(mytheta, myX)
    term1 = np.log(myh).dot(-myy.T)
    term2 = np.log(1.0 - myh).dot(1 - myy.T)
    left_hand = (term1 - term2) / m
    right_hand = mytheta.T.dot(mytheta) * mylambda / (2 * m)
    return left_hand + right_hand


def costGradient(mytheta, myX, myy, mylambda = 0.):
    m = myX.shape[0]

    beta = h(mytheta, myX) - myy.T

    regterm = mytheta[1:] * (mylambda / m)

    grad = (1./m)*np.dot(myX.T,beta)

    grad[1:] = grad[1:] + regterm
    return grad

def optimizeTheta(mytheta, myX, myy, mylambda = 0.):
    result = optimize.fmin_cg(computeCost, fprime = costGradient, x0 = mytheta, args = (myX, myy, mylambda), maxiter = 50, disp=False, full_output=True)
    return result[0], result[1]

def buildTheta(X, y):
    mylambda = 0.
    initial_theta = np.zeros((X.shape[1], 1)).reshape(-1)
    Theta = np.zeros((10, X.shape[1]))
    for i in range(10):
        iclass = i if i else 10;
        print("Optimizing for number %d..." % iclass);
        logic_Y = np.array([1 if x == iclass else 0 for x in y])
        itheta, imincost = optimizeTheta(initial_theta, X, logic_Y, mylambda)
    print("Done!")
    return Theta


def predictOneVsAll(mytheta, myrow):
    classes = [10] + range(1, 10)
    hypots = [0] * len(classes)
    for i in range(len(classes)):
        hypots[i] = h(mytheta[i], myrow)
    return classes[np.argmax(np.array(hypots))]