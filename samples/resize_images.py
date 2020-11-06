

import skimage.io
import skimage.transform
import os

dir = "C:\\Users\\lukas\\Dropbox\\studium\\Masterarbeit\\img\\noise"

for file in os.listdir(dir):
    filename = dir + "/" + file
    if os.path.isfile(filename):
        img = skimage.io.imread(filename)
        resized = skimage.transform.resize(image=img, output_shape=(360,640))
        skimage.io.imsave(dir + "/" + "resized/" + file, resized)
