from collections import OrderedDict

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from hyperopt.plotting import main_plot_history, main_plot_histogram, main_plot_vars

space = OrderedDict([
    ('backbone', ["resnet50_batch_size1", "resnet50_batch_size2", "resnet101"]),
    ('train_rois_per_image', [50, 100, 200]),
    ('detection_min_confidence', [0.6, 0.7, 0.8]),
    ('optimizer', ['ADAM', 'SGD'])
])


def scatter_plot(data_x, data_y, x_label, y_label):
    font = {'family': 'arial',
            'size': 16}

    matplotlib.rc('font', **font)

    colors = (0, 0, 0)
    area = np.pi * 10
    plt.scatter(data_x, data_y, s=area, c=colors, alpha=0.8)
    plt.xticks(np.unique(data_x))
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


def plot_sun_rgb_results(result):
    scatter_plot(result['detection_min_confidence'], result['f1score'], 'detection min confidence',
                 'f1 score')
    scatter_plot(result['optimizer'], result['f1score'], 'optimizer',
                 'f1 score')
    scatter_plot(result['backbone'], result['f1score'], 'backbone',
                 'f1 score')
    scatter_plot(result['train_rois_per_image'], result['f1score'], 'train_rois_per_image',
                 'f1 score')


def plot_hyperopt():
    trials = joblib.load('C:\\Users\\p41929\\_Master Thesis\\Mask_RCNN\\logs\\hyperopt_trials_9.pkl')
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
    plot_sun_rgb_results(result)


plot_hyperopt()
