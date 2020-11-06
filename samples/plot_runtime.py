import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

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


FILE_DIR = os.path.abspath("C:\\Users\\lukas\\Dropbox\\studium\\Masterarbeit\\src\\")
runtime_rgb = np.load(FILE_DIR + "/runtimes_CPU_ELEVATOR_RGB.npy").clip(min=0)
runtime_d3 = np.load(FILE_DIR + "/runtimes_CPU_ELEVATOR_D3.npy").clip(min=0)
runtime_rgbd = np.load(FILE_DIR + "/runtimes_CPU_ELEVATOR_RGBD.npy").clip(min=0)
runtime_rgbdf = np.load(FILE_DIR + "/runtimes_CPU_ELEVATOR_RGBDFusenet.npy").clip(min=0)

print(runtime_rgb.mean())
print(runtime_rgb.std())
print(runtime_d3.mean())
print(runtime_d3.std())
print(runtime_rgbd.mean())
print(runtime_rgbd.std())
print(runtime_rgbdf.mean())
print(runtime_rgbdf.std())

print(runtime_rgb.shape)

fig, ax = plt.subplots()

data = pd.DataFrame()
data["RGB"] = runtime_rgb
data["D3"] = runtime_d3
data["RGBD"] = runtime_rgbd
data["RGBD-F"] = runtime_rgbdf
data.boxplot()
print(data.shape)
#ax.boxplot(data)


ax.set_xlabel("model-version")
ax.set_ylabel("inference time")
# ax.set_yscale("log")
plt.show()
