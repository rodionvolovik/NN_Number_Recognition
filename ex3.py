import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io#Used to load the OCTAVE *.mat files
import scipy.misc#Used to show matrix as an image
import matplotlib.cm as cm#Used to display images in a specific colormap
import random#To pick random images to display
from scipy.special import expit#Vectorized sigmoid function

from visu import *
from algo import *

# 1. Multiclass classification
# 
#   1.1. Dataset 

data = scipy.io.loadmat('ex3data1.mat')
X, y = data['X'], data['y']
X = np.insert(X, 0, 1, axis = 1) #insert a column of 1's to X

#   1.2. Visualizing data 
displayData(X)

#   1.3. Vectorising logistic regression
#   1.4 One-vs-all Classification

Theta = buildTheta(X, y) 

n_correct, n_total = 0., 0.
incorrect_indicies = []
for irow in range(X.shape[0]):
    n_total += 1
    if predictOneVsAll(Theta, X[irow]) == y[irow]:
        n_correct += 1
    else:
        incorrect_indicies.append(irow)
print("Training set accuracy: %0.1f%%" % (100 * (n_correct/n_total)))




#already trained by us. These are stored in ex3weights.mat
datafile = 'ex3weights.mat'
mat = scipy.io.loadmat( datafile )
Theta1, Theta2 = mat['Theta1'], mat['Theta2']

def propagateForward(row,Thetas):
    """
    Function that given a list of Thetas, propagates the
    Row of features forwards, assuming the features already
    include the bias unit in the input layer, and the 
    Thetas need the bias unit added to features between each layer
    """
    features = row
    for i in xrange(len(Thetas)):
        Theta = Thetas[i]
        z = Theta.dot(features)
        a = expit(z)
        if i == len(Thetas)-1:
            return a
        a = np.insert(a,0,1) #Add the bias unit
        features = a

def predictNN(row,Thetas):
    """
    Function that takes a row of features, propagates them through the
    NN, and returns the predicted integer that was hand written
    """
    classes = range(1,10) + [10]
    output = propagateForward(row,Thetas)
    return classes[np.argmax(np.array(output))]

# "You should see that the accuracy is about 97.5%"
myThetas = [ Theta1, Theta2 ]
n_correct, n_total = 0., 0.
incorrect_indices = []
#Loop over all of the rows in X (all of the handwritten images)
#and predict what digit is written. Check if it's correct, and
#compute an efficiency.
for irow in xrange(X.shape[0]):
    n_total += 1
    if predictNN(X[irow],myThetas) == int(y[irow]): 
        n_correct += 1
    else: incorrect_indices.append(irow)
print("Training set accuracy: %0.1f%%" % (100 * (n_correct / n_total)))

#Pick some of the images we got WRONG and look at them, just to see
for x in xrange(5):
    i = random.choice(incorrect_indices)
    fig = plt.figure(figsize=(3,3))
    img = scipy.misc.toimage( getDatumImg(X[i]) )
    plt.imshow(img,cmap = cm.Greys_r)
    predicted_val = predictNN(X[i],myThetas)
    predicted_val = 0 if predicted_val == 10 else predicted_val
    fig.suptitle('Predicted: %d'%predicted_val, fontsize=14, fontweight='bold')
    plt.show()
