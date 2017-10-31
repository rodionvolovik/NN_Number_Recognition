import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io#Used to load the OCTAVE *.mat files
import scipy.misc#Used to show matrix as an image
import matplotlib.cm as cm#Used to display images in a specific colormap
import random#To pick random images to display
from scipy.special import expit#Vectorized sigmoid function

from visu import *

def getDatumImg(row):
    width, height = 20, 20
    square = row[1:].reshape(width, height)
    return square.T


def displayData(X, indicies_to_display = None):
    width, height = 20, 20
    nrows, ncols = 10, 10
    if not indicies_to_display:
        indicies_to_display = random.sample(range(X.shape[0]), nrows*ncols)

    big_picture = np.zeros((height*nrows, width*ncols))

    irow, icol = 0, 0
    for idx in indicies_to_display:
        if icol == ncols:
            irow += 1
            icol = 0

        ith_img = getDatumImg(X[idx])

        big_picture[irow * height:irow * height + ith_img.shape[0], icol * width:icol * width + ith_img.shape[1]] = ith_img
        icol += 1

    fig = plt.figure(figsize=(6,6))
    img = scipy.misc.toimage(big_picture)
    plt.imshow(img,cmap = cm.Greys_r)
    plt.show()
