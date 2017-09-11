##############################################
#
# change image size and save to another image
#
###############################################

IN_FOLDER="/home/junwon/tensorflow_code/BEGAN-tensorflow/data/R0402"
OUT_FOLDER="/home/junwon/tensorflow_code/BEGAN-tensorflow/data/R0402_212d"

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import cv2

if not os.path.exists(OUT_FOLDER):
    os.mkdir(OUT_FOLDER)



def main():
    for filename in os.listdir(IN_FOLDER):
        f_path = os.path.join(IN_FOLDER,filename)
        print filename
        img = mpimg.imread(f_path)
        print "before:",img.shape
        img = cv2.resize(img,(212,212))
        print "after:",img.shape
        cv2.imwrite(os.path.join(OUT_FOLDER,filename),img)

main()

