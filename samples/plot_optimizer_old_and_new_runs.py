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

sgd_old_tra = pd.read_csv(
    "C:\\Users\\lukas\\Dropbox\\studium\\Masterarbeit\\src\\training\\optimizer_old\\run-sunrgb20200704T1912_train-tag-epoch_loss.csv")
sgd_old_val = pd.read_csv(
    "C:\\Users\\lukas\\Dropbox\\studium\\Masterarbeit\\src\\training\\optimizer_old\\run-sunrgb20200704T1912_validation-tag-epoch_loss.csv")
adam_tra = pd.read_csv(
    "C:\\Users\\lukas\\Dropbox\\studium\\Masterarbeit\\src\\training\\optimizer_old\\run-additional_sunrgb20200706T1041_train-tag-epoch_loss.csv")
adam_val = pd.read_csv(
    "C:\\Users\\lukas\\Dropbox\\studium\\Masterarbeit\\src\\training\\optimizer_old\\run-additional_sunrgb20200706T1041_validation-tag-epoch_loss.csv")
adam_new_tra = pd.read_csv(
    "C:\\Users\\lukas\\Dropbox\\studium\\Masterarbeit\\src\\training\\optimizer\\run-sunrgb20200814T2139_train-tag-epoch_loss.csv")
adam_new_val = pd.read_csv(
    "C:\\Users\\lukas\\Dropbox\\studium\\Masterarbeit\\src\\training\\optimizer\\run-sunrgb20200814T2139_validation-tag-epoch_loss.csv")
sgd_tra = pd.read_csv(
    "C:\\Users\\lukas\\Dropbox\\studium\\Masterarbeit\\src\\training\\run-sunrgb20200811T2232_train-tag-epoch_loss.csv")
sgd_val = pd.read_csv(
    "C:\\Users\\lukas\\Dropbox\\studium\\Masterarbeit\\src\\training\\run-sunrgb20200811T2232_validation-tag-epoch_loss.csv")

sgd_old_tra = sgd_old_tra.head(264)
sgd_old_val = sgd_old_val.head(264)

sgd_old_tra = sgd_old_tra.groupby(np.arange(len(sgd_old_tra)) // 6).mean()
sgd_old_val = sgd_old_val.groupby(np.arange(len(sgd_old_val)) // 6).mean()
adam_tra = adam_tra.groupby(np.arange(len(adam_tra)) // 6).mean()
adam_val = adam_val.groupby(np.arange(len(adam_val)) // 6).mean()
print(sgd_old_tra)
print(sgd_old_val)


training = pd.DataFrame(columns=["SGD", "ADAM"])
training["SGD"] = sgd_old_tra["Value"]
training["ADAM"] = adam_tra["Value"]
training["ADAM new"] = adam_new_tra["Value"]
training["SGD new"] = sgd_tra["Value"]

training.plot()

print(training)

plt.ylabel("training loss")
plt.xlabel("epoch")
plt.show()
validation = pd.DataFrame(columns=["SGD", "ADAM"])
validation["SGD"] = sgd_old_val["Value"]
validation["ADAM"] = adam_val["Value"]
validation["ADAM new"] = adam_new_val["Value"]
validation["SGD new"] = sgd_val["Value"]

validation['index'] = range(0, len(validation))

validation_moving = validation.iloc[:, :].rolling(window=10).mean()
print(validation_moving)
validation_moving["index"] = validation['index']

x = validation["index"]
fig, ax = plt.subplots()

ax.plot(validation_moving["index"], validation_moving["SGD"], "-", label="SGD")
ax.plot(validation_moving["index"], validation_moving["SGD new"], "-", label="SGD new")
ax.plot(validation_moving["index"], validation_moving["ADAM"], "-", label="ADAM")
ax.plot(validation_moving["index"], validation_moving["ADAM new"], "-", label="ADAM new")


ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
          ncol=3, fancybox=True, shadow=False)
ax.set_xlabel("epochs")
ax.set_ylabel("validation loss")
# ax.set_yscale("log")
plt.show()
