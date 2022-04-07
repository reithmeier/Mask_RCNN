# **********************************************************************************************************************
#
# brief:    simple script to plot the optimizer runs
#
# author:   Lukas Reithmeier
# date:     25.08.2020
#
# **********************************************************************************************************************


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('pgf')
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


rgb_tra = pd.read_csv(
    "run-elevator_rgb_train-tag-epoch_loss.csv")
rgb_val = pd.read_csv(
    "run-elevator_rgb_validation-tag-epoch_loss.csv")
d3_tra = pd.read_csv(
    "run-elevator_d3_train-tag-epoch_loss.csv")
d3_val = pd.read_csv(
    "run-elevator_d3_validation-tag-epoch_loss.csv")
rgbd_tra = pd.read_csv(
    "run-elevator_rgbd_train-tag-epoch_loss.csv")
rgbd_val = pd.read_csv(
    "run-elevator_rgbd_validation-tag-epoch_loss.csv")
rgbdf_tra = pd.read_csv(
    "run-elevator_rgbd_fusenet_train-tag-epoch_loss.csv")
rgbdf_val = pd.read_csv(
    "run-elevator_rgbd_fusenet_validation-tag-epoch_loss.csv")

print(rgb_tra)
print(rgb_val)

training = pd.DataFrame(columns=["RGB", "D3", "RGBD", "RGBD-F"])
training["RGB"] = rgb_tra["Value"]
training["D3"] = d3_tra["Value"]
training["RGBD"] = rgbd_tra["Value"]
training["RGBD-F"] = rgbdf_tra["Value"]
training.plot()

print(training)

plt.ylabel("training loss")
plt.xlabel("epoch")
tikz_save("elevator_runs_train.tex")
plt.show()

validation = pd.DataFrame(columns=["RGB", "D3", "RGBD", "RGBD-F"])
validation["RGB"] = rgb_val["Value"]
validation["D3"] = d3_val["Value"]
validation["RGBD"] = rgbd_val["Value"]
validation["RGBD-F"] = rgbdf_val["Value"]
validation['index'] = range(0, len(validation))

rgb_trend = get_trend(validation["index"], validation["RGB"])
d3_trend = get_trend(validation["index"], validation["D3"])
rgbd_trend = get_trend(validation["index"], validation["RGBD"])
rgbdf_trend = get_trend(validation["index"], validation["RGBD-F"])

validation_avg = validation.groupby(np.arange(len(validation)) // 5).mean()

validation_moving = validation.iloc[:, :].rolling(window=10).mean()
print(validation_moving)
validation_moving["index"] = validation['index']

x = validation["index"]
fig, ax = plt.subplots()


ax.plot(validation_moving["index"], validation_moving["RGB"], "-", label="avg RGB")
ax.plot(validation_moving["index"], validation_moving["D3"], "-", label="avg D3")
ax.plot(validation_moving["index"], validation_moving["RGBD"], "-", label="avg RGBD")
ax.plot(validation_moving["index"], validation_moving["RGBD-F"], "-", label="avg RGBD-F")


ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
          ncol=3, fancybox=True, shadow=False)
ax.set_xlabel("epochs")
ax.set_ylabel("validation loss")
tikz_save("elevator_runs_val.tex")
plt.show()
