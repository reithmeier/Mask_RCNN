from matplotlib import pyplot as plt
import pandas as pd
from tikzplotlib import save as tikz_save
import matplotlib
#matplotlib.use('pgf')

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
plt.style.use('ggplot')


def plot_precision_recall_curve(precissions, recalls):
    fig, ax = plt.subplots()
    ax.plot(recalls, precissions, label="RGB")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precission")
    ax.set_xlim(0., 1.1)
    ax.set_ylim(0., 1.1)
    plt.show()


def plot_precision_recall_curves(precissions_rgb, recalls_rgb, precisions_d3, recalls_d3, precisions_rgbd, recalls_rgbd,
                                 precisions_rgbdf, recalls_rgbdf):
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(recalls_rgb, precissions_rgb, label="RGB")
    ax.plot(recalls_d3, precisions_d3, label="D3")
    ax.plot(recalls_rgbd, precisions_rgbd, label="RGBD")
    ax.plot(recalls_rgbdf, precisions_rgbdf, label="RGBD-F")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precission")
    #ax.set_xlim(0., 1.)
    ax.set_ylim(0., 1.05)
    ax.legend()
    tikz_save("precision_recall_elevator.tex")

    plt.show()


data_rgb = pd.read_csv("./precision_recalls_ELEVATOR_RGB.csv")
plot_precision_recall_curve(data_rgb["precision"], data_rgb["recall"])
data_d3 = pd.read_csv("./precision_recalls_ELEVATOR_D3.csv")
plot_precision_recall_curve(data_d3["precision"], data_d3["recall"])
data_rgbd = pd.read_csv("./precision_recalls_ELEVATOR_RGBD.csv")
plot_precision_recall_curve(data_rgbd["precision"], data_rgbd["recall"])
data_rgbdf = pd.read_csv("./precision_recalls_ELEVATOR_RGBDFusenet.csv")
plot_precision_recall_curve(data_rgbdf["precision"], data_rgbdf["recall"])

plot_precision_recall_curves(data_rgb["precision"], data_rgb["recall"], data_d3["precision"], data_d3["recall"],
                             data_rgbd["precision"], data_rgbd["recall"],
                             data_rgbdf["precision"], data_rgbdf["recall"])
