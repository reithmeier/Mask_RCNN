# **********************************************************************************************************************
#
# brief:    simple script to plot the optimizer runs
#
# author:   Lukas Reithmeier
# date:     29.09.2020
#
# **********************************************************************************************************************


from matplotlib import pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl
import numpy as np


def draw_image(img):
    mpl.colors.Normalize(vmin=0., vmax=1.)

    fig, ax = plt.subplots()
    plt.axis('off')
    plt.margins = 0
    plt.imshow(img, cmap='viridis')
    # Loop over data dimensions and create text annotations.
    for i in range(8):
        for j in range(8):
            text = ax.text(j, i, img[i, j],
                           ha="center", va="center", color="w")
    plt.show()


def draw_image_with_roi(img, roi):
    mpl.colors.Normalize(vmin=0., vmax=1.)

    fig, ax = plt.subplots()
    plt.axis('off')
    plt.margins = 0
    plt.imshow(img, cmap='viridis')
    # Loop over data dimensions and create text annotations.
    for i in range(8):
        for j in range(8):
            text = ax.text(j, i, img[i, j],
                           ha="center", va="center", color="w")

    rect = patches.Rectangle((roi[0] - 0.5, roi[1] - 0.5), roi[2], roi[3], linewidth=3, edgecolor='r', facecolor='none')
    ax.add_patch(rect)

    plt.show()


def draw_image_with_roi_and_slices(img, roi, slices):
    mpl.colors.Normalize(vmin=0., vmax=1.)

    fig, ax = plt.subplots()
    plt.axis('off')
    plt.margins = 0
    plt.imshow(img, cmap='viridis')
    # Loop over data dimensions and create text annotations.
    for i in range(8):
        for j in range(8):
            text = ax.text(j, i, img[i, j],
                           ha="center", va="center", color="w")

    rect = patches.Rectangle((roi[0] - 0.5, roi[1] - 0.5), roi[2], roi[3], linewidth=3, edgecolor='r', facecolor='none')
    ax.add_patch(rect)

    for slice in slices:
        rect = patches.Rectangle((slice[0] - 0.5, slice[1] - 0.5), slice[2], slice[3], linewidth=3, edgecolor='r',
                                 facecolor='none')
        ax.add_patch(rect)

    plt.show()


def draw_pooled(img, slices):
    maxima = np.zeros((2, 2), np.float)
    sliced1 = img[slices[0][1]:slices[0][1] + slices[0][3], slices[0][0]:slices[0][0] + slices[0][2]]
    sliced2 = img[slices[1][1]:slices[1][1] + slices[1][3], slices[1][0]:slices[1][0] + slices[1][2]]
    sliced3 = img[slices[2][1]:slices[2][1] + slices[2][3], slices[2][0]:slices[2][0] + slices[2][2]]
    sliced4 = img[slices[3][1]:slices[3][1] + slices[3][3], slices[3][0]:slices[3][0] + slices[3][2]]

    print(sliced1)
    print(sliced2)
    print(sliced3)
    print(sliced4)

    maxima[0, 0] = np.max(sliced1)
    maxima[0, 1] = np.max(sliced2)
    maxima[1, 0] = np.max(sliced3)
    maxima[1, 1] = np.max(sliced4)

    print(maxima)
    mpl.colors.Normalize(vmin=-1., vmax=1.)
    fig, ax = plt.subplots()
    plt.axis('off')
    plt.margins = 0
    plt.imshow(np.array(maxima), cmap='viridis')
    # Loop over data dimensions and create text annotations.
    for i in range(2):
        for j in range(2):
            text = ax.text(j, i, maxima[i, j],
                           ha="center", va="center", color="w")
    ax.add_patch(patches.Rectangle((-0.5, - 0.5), 1, 1, linewidth=3, edgecolor='r', facecolor='none'))
    ax.add_patch(patches.Rectangle((0.5, - 0.5), 1, 1, linewidth=3, edgecolor='r', facecolor='none'))
    ax.add_patch(patches.Rectangle((-0.5, 0.5), 1, 1, linewidth=3, edgecolor='r', facecolor='none'))
    ax.add_patch(patches.Rectangle((0.5, 0.5), 1, 1, linewidth=3, edgecolor='r', facecolor='none'))

    plt.show()


img = np.round(np.random.rand(8, 8) * 10) / 10
roi = [0, 3, 7, 5]
slices = [
    [0, 3, 3, 2],
    [3, 3, 4, 2],
    [0, 5, 3, 3],
    [3, 5, 4, 4]
]

print(img)

draw_image(img)
draw_image_with_roi(img, roi)
draw_image_with_roi_and_slices(img, roi, slices)
draw_pooled(img, slices)
