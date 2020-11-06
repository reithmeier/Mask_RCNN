# **********************************************************************************************************************
#
# brief:    simple script to plot the hyper parameter optimization results
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


# search space
space = OrderedDict([
    ('num_filters', ["[32, 32, 64, 128, 256]", "[64, 64, 128, 256, 512]", "[128, 128, 256, 512, 1024]"]),
    ('backbone', ["resnet50_batch_size1", "resnet50_batch_size2", "resnet101"]),
    ('train_rois_per_image', [50, 100, 200]),
    ('detection_min_confidence', [0.6, 0.7, 0.8]),
    ('optimizer', ['ADAM', 'SGD'])
])


def scatter_plot(data_x, data_y, x_label, y_label):
    """
    print a scatter plot
    :param data_x: data in x direction
    :param data_y: data in y direction
    :param x_label: label of x axis
    :param y_label: label of y axis
    :return:
    """

    font = {'family': 'arial',
            'size': 16}

    matplotlib.rc('font', **font)

    colors = np.arange(10)
    area = np.pi * 30

    plt.scatter(data_x, data_y, s=area, c=colors, alpha=0.8)
    plt.set_cmap(matplotlib.cm.get_cmap('tab10'))
    plt.margins(0.2, 0.1)

    plt.xticks(np.unique(data_x))
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


def wrap_labels(labels):
    for i in range(len(labels)):
        if isinstance(labels[i], str):
            labels[i] = labels[i].replace('_', '\n', 1)
            labels[i] = labels[i].replace('_', ' ')
    return labels


def plot_results(result):
    """
    plot results of a sun rgb hyperparameter optimization
    :param result: results to be plotter
    """
    scatter_plot(result['detection_min_confidence'], result['f1score'], 'detection min confidence',
                 'f1 score')
    scatter_plot(result['optimizer'], result['f1score'], 'optimizer',
                 'f1 score')
    if 'backbone' in result:
        scatter_plot(wrap_labels(result['backbone']), result['f1score'], 'backbone',
                     'f1 score')
    if 'num_filters' in result:
        result['num_filters'] = np.where(result['num_filters'] == '[32, 32, 64, 128, 256]', 'small',
                                         result['num_filters'])
        result['num_filters'] = np.where(result['num_filters'] == '[64, 64, 128, 256, 512]', 'medium',
                                         result['num_filters'])
        result['num_filters'] = np.where(result['num_filters'] == '[128, 128, 256, 512, 1024]', 'large',
                                         result['num_filters'])
        result['num_filters'] = np.where(result['num_filters'] == '[32, 32, 64, 128, 256]', 'small',
                                         result['num_filters'])
        scatter_plot(result['num_filters'], result['f1score'], 'num filters',
                     'f1 score')
    scatter_plot(result['train_rois_per_image'], result['f1score'], 'train rois per image',
                 'f1 score')


def plot_correlation(corr):
    sns.heatmap(corr, annot=True, cmap=sns.diverging_palette(20, 220, n=256), annot_kws={"size": SMALL_SIZE})
    plt.show()


def plot_hyperopt(trials_file, time_file, res_file):
    """
    load trials file and plot all results
    :param trials_file: path of the trials.pkl file
    """
    trials = joblib.load(trials_file)
    main_plot_history(trials=trials)
    main_plot_histogram(trials=trials)
    main_plot_vars(trials=trials)

    # parse trials object
    result = {'f1score': []}
    i = 0
    for configuration in trials.miscs:
        for hyperparam in configuration['vals']:
            idx = configuration['vals'][hyperparam][0]
            if hyperparam not in result:
                result[hyperparam] = []
            result[hyperparam].append(space[hyperparam][idx])
        result['f1score'].append(-1 * trials.results[i]['loss'])
        i += 1
    print(result)
    # plot results
    plot_results(result)

    times = pd.read_csv(time_file, delimiter=' ', header=None)
    times.columns = ['h', 'm', 's']
    print(times)
    times['time'] = times['h'] + times['m'] / 60 + times['s'] / 60 / 60
    times = times.drop(['h', 'm', 's'], axis=1)
    print(times)
    results = pd.DataFrame(result)
    print(results)

    result = pd.concat([results, times], axis=1)
    print(result)

    print(result)
    result.to_csv(res_file, index=False)
    tokenized_result = result

    tokenize_bb = lambda s: 0 if s == "resnet101" else (1 if s == "resnet50\nbatch size1" else 2)
    tokenize_opt = lambda s: 0 if s == "SGD" else 1
    tokenized_result['tok_backbone'] = tokenized_result['backbone'].apply(tokenize_bb)
    tokenized_result['tok_optimizer'] = tokenized_result['optimizer'].apply(tokenize_opt)
    tokenized_result = tokenized_result.drop(['backbone', 'optimizer'], axis=1)
    tokenized_result = tokenized_result.rename(columns={
        'tok_backbone': 'backbone',
        'tok_optimizer': 'optimizer',
        'detection_min_confidence': 'dmc',
        'train_rois_per_image': 'rois'
    })
    print(tokenized_result)
    correlation = tokenized_result.corr(method='pearson')
    focus_cols = ['f1score', 'time']
    correlation = correlation.filter(focus_cols).drop(focus_cols)

    print(correlation)
    correlation.to_csv(res_file + ".corr.csv", index=False)

    plot_correlation(correlation)


# plot_hyperopt(ROOT_DIR + '/logs/hyperopt_sund3/hyperopt_trials_9.pkl')

plot_hyperopt(ROOT_DIR +'/logs/hparam_tuning_sunrgbd\\hyperopt_trials_9.pkl',
              'C:\\Users\\lukas\\Dropbox\\studium\\Masterarbeit\\src\\hyperopt_time_rgbd.txt',
              'C:\\Users\\lukas\\Dropbox\\studium\\Masterarbeit\\src\\hyperopt_time_rgbd_res.csv')
