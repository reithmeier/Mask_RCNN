# **********************************************************************************************************************
#
# brief:    script to print the distribution of classes in the data set
#
# author:   Lukas Reithmeier
# date:     23.04.2020
#
# **********************************************************************************************************************


import argparse
import json
import os
import numpy as np
from matplotlib import pyplot as plt

ROOT_DIR = os.path.abspath("./../../..")
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


def main():
    class_cnt = {}
    labels_per_file = {}
    for filename in os.listdir(args.input):
        if filename.endswith(".json"):
            with open(args.input + "/" + filename) as in_file:
                data = json.load(in_file)
                annotations = data['completions'][-1]["result"]
                label_cnt = 0
                for ann in annotations:
                    if ann["type"] == "polygonlabels":
                        label_cnt = label_cnt + 1
                        label = ann["value"]["polygonlabels"][0]
                        if label in class_cnt:
                            class_cnt[label] = class_cnt[label] + 1
                        else:
                            class_cnt[label] = 1

                if label_cnt in labels_per_file:
                    labels_per_file[label_cnt] = labels_per_file[label_cnt] + 1
                else:
                    labels_per_file[label_cnt] = 1

    print('labels')
    for key in class_cnt:
        print(key, '->', class_cnt[key])

    print('\nlabels per file')
    for key in labels_per_file:
        print(str(key), '->', labels_per_file[key])

    with plt.style.context('ggplot'):
        fig, ax = plt.subplots()
        keys = labels_per_file.keys()
        values = labels_per_file.values()
        ax.set_yscale("log")
        plt.xticks(np.arange(min(keys), max(keys) + 1, 1.0))

        rects = ax.bar(keys, values, edgecolor="w", linewidth=1, color="CornFlowerBlue")
        ax.set_xlabel('Number of samples', fontsize=MEDIUM_SIZE)
        ax.set_ylabel('Labels per sample', fontsize=MEDIUM_SIZE)
        autolabel(rects, ax)
    from tikzplotlib import save as tikz_save
    tikz_save("elevator_dataset_labels_per_image.tex")
    plt.show()

    with plt.style.context('ggplot'):
        plt.style.use('ggplot')
        fig, ax = plt.subplots()
        keys = class_cnt.keys()
        values = class_cnt.values()
        ax.set_yscale("log")
        plt.xticks(rotation=90)
        rects = ax.bar(keys, values, edgecolor="w", linewidth=1, color="CornFlowerBlue")
        ax.set_ylabel('Number of objects', fontsize=MEDIUM_SIZE)
        ax.set_xlabel('Class', fontsize=MEDIUM_SIZE)

        autolabel(rects, ax)
    tikz_save("elevator_dataset_objects_per_class.tex")
    plt.show()


def autolabel(rects, ax):
    plt.style.use('ggplot')

    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', color="dimgray", fontsize=SMALL_SIZE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, help="Path of the input directory",
                        default=ROOT_DIR + "/datasets/elevator/preprocessed/labels/")
    args = parser.parse_args()
    main()
