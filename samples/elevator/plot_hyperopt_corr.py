
from collections import OrderedDict
import os
import joblib
import matplotlib
import matplotlib.cm
import matplotlib.pyplot as plt
import numpy as np
from hyperopt.plotting import main_plot_history, main_plot_histogram, main_plot_vars
import pandas as pd
import seaborn as sns

ROOT_DIR = os.path.abspath("../")
plt.style.use('ggplot')

SMALL_SIZE = 20
MEDIUM_SIZE = 22
BIGGER_SIZE = 24

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


def plot_correlation(corr, ver, bb):
    corr.index = ['dmc', 'rois', bb, 'optimizer']

    print(corr)
    sns.heatmap(corr, annot=True, cmap=sns.diverging_palette(20, 220, n=256), annot_kws={"size": SMALL_SIZE})

    #plt.show()
    from tikzplotlib import save as tikz_save
    tikz_save(f"hyperopt_{ver}_corr.tex")


corr_rgb = 'C:\\Users\\lukas\\Dropbox\\studium\\Masterarbeit\\src\\hyperopt_time_rgb_res.csv.corr.csv'
corr_d3 = 'C:\\Users\\lukas\\Dropbox\\studium\\Masterarbeit\\src\\hyperopt_time_d3_res.csv.corr.csv'
corr_rgbd = 'C:\\Users\\lukas\\Dropbox\\studium\\Masterarbeit\\src\\hyperopt_time_rgbd_res.csv.corr.csv'
corr_rgbdf = 'C:\\Users\\lukas\\Dropbox\\studium\\Masterarbeit\\src\\hyperopt_time_rgbd_fusenet_res.csv.corr.csv'
corr_rgb = pd.read_csv(corr_rgb)
corr_d3 = pd.read_csv(corr_d3)
corr_rgbd = pd.read_csv(corr_rgbd)
corr_rgbdf = pd.read_csv(corr_rgbdf)

plot_correlation(corr_rgb, "rgb", "backbone")
plot_correlation(corr_d3, "d3", "backbone")
plot_correlation(corr_rgbd, "rgbd", "backbone")
plot_correlation(corr_rgbdf, "rgbd_fusenet", "filters")
