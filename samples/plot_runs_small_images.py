import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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


big_tra = pd.read_csv(
    "C:\\Users\\lukas\\Dropbox\\studium\\Masterarbeit\\src\\training\\run-sunrgb20200811T2232_train-tag-epoch_loss.csv")
big_val = pd.read_csv(
    "C:\\Users\\lukas\\Dropbox\\studium\\Masterarbeit\\src\\training\\run-sunrgb20200811T2232_validation-tag-epoch_loss.csv")
sml_tra = pd.read_csv(
    "C:\\Users\\lukas\\Dropbox\\studium\\Masterarbeit\\src\\training\\small_images\\run-sunrgb20200816T0246_train-tag-epoch_loss.csv")
sml_val = pd.read_csv(
    "C:\\Users\\lukas\\Dropbox\\studium\\Masterarbeit\\src\\training\\small_images\\run-sunrgb20200816T0246_validation-tag-epoch_loss.csv")


print(big_tra)
print(big_val)

training = pd.DataFrame(columns=["512 x 512", "256 x 256"])
training["512 x 512"] = big_tra["Value"]
training["256 x 256"] = sml_tra["Value"]
training.plot()

print(training)

plt.ylabel("training loss")
plt.xlabel("epoch")
plt.show()
validation = pd.DataFrame(columns=["512 x 512", "256 x 256"])
validation["512 x 512"] = big_val["Value"]
validation["256 x 256"] = sml_val["Value"]
validation['index'] = range(0, len(validation))

rgb_trend = get_trend(validation["index"], validation["512 x 512"])
d3_trend = get_trend(validation["index"], validation["256 x 256"])

validation_avg = validation.groupby(np.arange(len(validation)) // 5).mean()
# validation_avg["index"] = [5, 15, 25, 35, 45]

validation_moving = validation.iloc[:, :].rolling(window=10).mean()
print(validation_moving)
validation_moving["index"] = validation['index']

x = validation["index"]
fig, ax = plt.subplots()
# ax.plot(x, validation["RGB"], label="RGB")
# ax.plot(x, validation["D3"], label="D3")
# ax.plot(x, validation["RGBD"], label="RGBD")
# ax.plot(x, validation["RGBD-F"], label="RGBD-F")


ax.plot(validation_moving["index"], validation_moving["512 x 512"], "-", label="512 x 512")
ax.plot(validation_moving["index"], validation_moving["256 x 256"], "-", label="256 x 256")


#ax.plot(validation_avg["index"], validation_avg["RGB"], "-", label="avg RGB")
#ax.plot(validation_avg["index"], validation_avg["D3"], "-", label="avg D3")
#ax.plot(validation_avg["index"], validation_avg["RGBD"], "-", label="avg RGBD")
#ax.plot(validation_avg["index"], validation_avg["RGBD-F"], "-", label="avg RGBD-F")

# ax.plot(validation["index"], rgb_trend(validation["index"]), "--", label="trend RGB")
# ax.plot(validation["index"], d3_trend(validation["index"]), "--", label="trend D3")
# ax.plot(validation["index"], rgbd_trend(validation["index"]), "--", label="trend RGBD")
# ax.plot(validation["index"], rgbdf_trend(validation["index"]), "--", label="trend RGBD-F")

ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
          ncol=3, fancybox=True, shadow=False)
ax.set_xlabel("epochs")
ax.set_ylabel("validation loss")
# ax.set_yscale("log")
plt.show()
