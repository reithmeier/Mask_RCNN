

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tikzplotlib import save as tikz_save


plt.style.use('ggplot')


SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 18

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


def get_trend(x, y):
    z = np.polyfit(x, y, 3)
    p = np.poly1d(z)
    return p


sun = pd.read_csv("C:\\Users\\lukas\\Dropbox\\studium\\Masterarbeit\\src\\training\\run-elevator_rgb20200814T1255_train-tag-epoch_loss.csv")
coco = pd.read_csv("C:\\Users\\lukas\\Dropbox\\studium\\Masterarbeit\\src\\training\\pretraining\\run-elevator_rgb20200815T1221_train-tag-epoch_loss.csv")
none = pd.read_csv("C:\\Users\\lukas\\Dropbox\\studium\\Masterarbeit\\src\\training\\pretraining\\run-elevator_rgb20200815T0733_train-tag-epoch_loss.csv")



training = pd.DataFrame(columns=[ "no pre-training", "SUN RGB-D", "MS COCO"])
training["no pre-training"] = none["Value"]
training["SUN RGB-D"] = sun["Value"]
training["MS COCO"] = coco["Value"]
training.plot()
plt.ylabel("training loss")
plt.xlabel("epoch")
tikz_save("elevator_rgb_pretraining_runs.tex")
plt.show()


