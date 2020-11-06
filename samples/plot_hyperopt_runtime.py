# **********************************************************************************************************************
#
# brief:    simple script to plot the hyper parameter optimization runtime results
#
# author:   Lukas Reithmeier
# date:     22.06.2020
#
# **********************************************************************************************************************

from collections import OrderedDict
import os
import joblib
import matplotlib
import matplotlib.cm
import matplotlib.pyplot as plt
import numpy as np
from hyperopt.plotting import main_plot_history, main_plot_histogram, main_plot_vars
import pandas as pd

ROOT_DIR = os.path.abspath("../")
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


def plot_training(rgb_file, d3_file, rgbd_file, rgbd_fusenet_file):
    rgb_result = pd.read_csv(rgb_file)
    d3_result = pd.read_csv(d3_file)
    rgbd_result = pd.read_csv(rgbd_file)
    rgbd_fusenet_result = pd.read_csv(rgbd_fusenet_file)

    fig, ax = plt.subplots()
    print(rgb_result)
    ax.plot(rgb_result['f1score'], label="RGB")
    ax.plot(d3_result['f1score'], label="D3")
    ax.plot(rgbd_result['f1score'], label="RGBD")
    ax.plot(rgbd_fusenet_result['f1score'], label="RGBD-F")
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.3),
              ncol=3, fancybox=False, shadow=False)
    plt.xlabel("Evaluation run")
    plt.ylabel("F1 score")
    from tikzplotlib import save as tikz_save
    tikz_save(f"hyperopt_evaluation_runs.tex")

    plt.show()



def plot_runtimes(rgb_file, d3_file, rgbd_file, rgbd_fusenet_file):
    rgb_result = pd.read_csv(rgb_file)
    d3_result = pd.read_csv(d3_file)
    rgbd_result = pd.read_csv(rgbd_file)
    rgbd_fusenet_result = pd.read_csv(rgbd_fusenet_file)
    data = [rgb_result['time'],
            d3_result['time'],
            rgbd_result['time'],
            rgbd_fusenet_result['time']
            ]
    fig, ax = plt.subplots()
    flierprops = {'color': 'black', 'marker': 'x'}
    boxprops = {'color': 'black', 'linestyle': '-'}

    ax.boxplot(data, flierprops=flierprops, boxprops=boxprops)
    plt.xlabel("model-version")
    plt.ylabel("time [h]")
    from tikzplotlib import save as tikz_save
    tikz_save(f"hyperopt_runtime_boxplot.tex")

    plt.show()


plot_runtimes('C:\\Users\\lukas\\Dropbox\\studium\\Masterarbeit\\src\\hyperopt_time_rgb_res.csv',
              'C:\\Users\\lukas\\Dropbox\\studium\\Masterarbeit\\src\\hyperopt_time_d3_res.csv',
              'C:\\Users\\lukas\\Dropbox\\studium\\Masterarbeit\\src\\hyperopt_time_rgbd_res.csv',
              'C:\\Users\\lukas\\Dropbox\\studium\\Masterarbeit\\src\\hyperopt_time_rgbd_fusenet_res.csv')



plot_training('C:\\Users\\lukas\\Dropbox\\studium\\Masterarbeit\\src\\hyperopt_time_rgb_res.csv',
              'C:\\Users\\lukas\\Dropbox\\studium\\Masterarbeit\\src\\hyperopt_time_d3_res.csv',
              'C:\\Users\\lukas\\Dropbox\\studium\\Masterarbeit\\src\\hyperopt_time_rgbd_res.csv',
              'C:\\Users\\lukas\\Dropbox\\studium\\Masterarbeit\\src\\hyperopt_time_rgbd_fusenet_res.csv')