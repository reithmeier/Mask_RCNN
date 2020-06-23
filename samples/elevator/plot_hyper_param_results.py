import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from hyperopt.plotting import main_plot_histogram, main_plot_history, main_plot_vars

ADAM = 'ADAM'
SGD = 'SGD'
resnet50_batch_size2 = "resnet50\nbatch size = 2"
resnet50_batch_size1 = "resnet50\nbatch size = 1"
resnet101 = "resnet101"

sun_rbg_results = {
    'detection_min_confidence': [0.7, 0.6, 0.7, 0.7, 0.6, 0.8, 0.6, 0.7],
    'optimizer': [ADAM, ADAM, ADAM, ADAM, ADAM, ADAM, SGD, SGD],
    'train_rois_per_image': [50, 200, 100, 200, 50, 100, 200, 50],
    'backbone': [resnet50_batch_size2, resnet50_batch_size1, resnet50_batch_size2, resnet50_batch_size1,
                 resnet50_batch_size1, resnet50_batch_size1, resnet101, resnet101],
    'f1score': [0.32434, 0.29214, 0.32956, 0.30526, 0.29998, 0.30383, 0.29369, 0.29637]
}


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


def plot_sun_rgb_results():
    scatter_plot(sun_rbg_results['detection_min_confidence'], sun_rbg_results['f1score'], 'detection min confidence',
                 'f1 score')
    scatter_plot(sun_rbg_results['optimizer'], sun_rbg_results['f1score'], 'optimizer',
                 'f1 score')
    scatter_plot(sun_rbg_results['backbone'], sun_rbg_results['f1score'], 'detection min confidence',
                 'f1 score')
    scatter_plot(sun_rbg_results['train_rois_per_image'], sun_rbg_results['f1score'], 'detection min confidence',
                 'f1 score')


def plot_hyperopt():
    trials = joblib.load('C:\\Users\\p41929\\_Master Thesis\\Mask_RCNN\\logs\\hyperopt_trials_9.pkl')
    main_plot_history(trials=trials)
    main_plot_histogram(trials=trials)
    main_plot_vars(trials=trials)


plot_sun_rgb_results()
plot_hyperopt()
