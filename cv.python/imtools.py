import sys
sys.path.insert(0,'/home/junwon/Downloads/PIL-1.1.7/PIL')


import Image
from pylab import *
from numpy import *

def histeq(im,nbr_bins=256):
    imhist,bins = histogram(im.flatten(),nbr_bins,normed=True)
    cdf = imhist.cumsum()
    cdf = 255 * cdf / cdf[-1]

    im2 = interp(im.flatten(), bins[:-1],cdf)

    return im2.reshape(im.shape), cdf

