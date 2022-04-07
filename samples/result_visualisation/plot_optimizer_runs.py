# **********************************************************************************************************************
#
# brief:    simple script to plot the optimizer runs
#
# author:   Lukas Reithmeier
# date:     16.08.2020
#
# **********************************************************************************************************************


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

plt.style.use('ggplot')

SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 18

plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


def get_trend(x, y):
    z = np.polyfit(x, y, 3)
    p = np.poly1d(z)
    return p


sgd_tra = pd.read_csv(
    "run-sunrgb_train-tag-epoch_loss.csv")
sgd_val = pd.read_csv(
    "run-sunrgb_validation-tag-epoch_loss.csv")
adam_tra = pd.read_csv(
    "run-sunrgb_train-tag-epoch_loss.csv")
adam_val = pd.read_csv(
    "run-sunrgb_validation-tag-epoch_loss.csv")

print(sgd_tra)
print(sgd_val)

training = pd.DataFrame(columns=["SGD", "ADAM"])
training["SGD"] = sgd_tra["Value"]
training["ADAM"] = adam_tra["Value"]
training['index'] = range(0, len(training))
training = training.head(25)

training.plot()

print(training)

plt.ylabel("training loss")
plt.xlabel("epoch")
plt.show()
validation = pd.DataFrame(columns=["SGD", "ADAM"])
validation["SGD"] = sgd_val["Value"]
validation["ADAM"] = adam_val["Value"]
validation['index'] = range(0, len(validation))
print(validation)

validation = validation.head(25)

validation_moving = validation.iloc[:, :].rolling(window=5).mean()
print(validation_moving)
validation_moving["index"] = validation['index']

x = validation["index"]
fig, ax = plt.subplots()

ax.plot(validation_moving["index"], validation_moving["SGD"], "-", label="SGD validation")
ax.plot(validation_moving["index"], validation_moving["ADAM"], "-", label="ADAM validation")
ax.plot(training["index"], training["SGD"], "-", label="SGD training")
ax.plot(training["index"], training["ADAM"], "-", label="ADAM training")


ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
          ncol=3, fancybox=True, shadow=False)
ax.set_xlabel("epochs")
ax.set_ylabel("validation loss")
plt.show()
