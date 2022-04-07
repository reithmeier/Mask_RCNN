# **********************************************************************************************************************
#
# brief:    simple script to plot runtimes
#
# author:   Lukas Reithmeier
# date:     15.08.2020
#
# **********************************************************************************************************************


import skimage.io
import skimage.transform
import os

dir = "../../logs/noise"

for file in os.listdir(dir):
    filename = dir + "/" + file
    if os.path.isfile(filename):
        img = skimage.io.imread(filename)
        resized = skimage.transform.resize(image=img, output_shape=(360,640))
        skimage.io.imsave(dir + "/" + "resized/" + file, resized)
