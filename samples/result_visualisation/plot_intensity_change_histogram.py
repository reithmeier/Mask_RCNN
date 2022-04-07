# **********************************************************************************************************************
#
# brief:    simple script to plot the intensity change histogram
#
# author:   Lukas Reithmeier
# date:     25.08.2020
#
# **********************************************************************************************************************


import math
import os

import matplotlib.pyplot as plt
import numpy as np

import skimage.io

plt.style.use('ggplot')
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


def distance(hist_1, hist_2):
    diff = hist_1 - hist_2
    return math.sqrt((diff * diff).sum())


def calc_diff(image1, image2, bins=255):
    hist_1_r, bins = np.histogram(image1[0].ravel(), 256, [0, 256])
    hist_1_g, _ = np.histogram(image1[1].ravel(), 256, [0, 256])
    hist_1_b, _ = np.histogram(image1[2].ravel(), 256, [0, 256])
    hist_2_r, _ = np.histogram(image2[0].ravel(), 256, [0, 256])
    hist_2_g, _ = np.histogram(image2[1].ravel(), 256, [0, 256])
    hist_2_b, _ = np.histogram(image2[2].ravel(), 256, [0, 256])

    num_pixels_1 = (image1.shape[0] * image1.shape[1])
    num_pixels_2 = (image2.shape[0] * image2.shape[1])
    hist_2_r = hist_2_r / num_pixels_2 * num_pixels_1
    hist_2_g = hist_2_g / num_pixels_2 * num_pixels_1
    hist_2_b = hist_2_b / num_pixels_2 * num_pixels_1

    max_r = max(hist_1_r.max(), hist_2_r.max())
    max_g = max(hist_1_g.max(), hist_2_g.max())
    max_b = max(hist_1_b.max(), hist_2_b.max())
    max_total = max(max_r, max_g, max_b)

    distance_r = distance(hist_1_r, hist_2_r)
    distance_g = distance(hist_1_g, hist_2_g)
    distance_b = distance(hist_1_b, hist_2_b)

    print(distance_r)
    print(distance_g)
    print(distance_b)
    return distance_r, distance_g, distance_b, max_total


def plot_hist_separate(image, max_total, bins=255, ver=""):
    fig, (ax_r, ax_b, ax_g) = plt.subplots(1, 3)

    hist_r, bins = np.histogram(image[0].ravel(), 256, [0, 256])
    hist_g, _ = np.histogram(image[1].ravel(), 256, [0, 256])
    hist_b, _ = np.histogram(image[2].ravel(), 256, [0, 256])
    num_pixels = (image.shape[0] * image.shape[1])

    ax_r.plot(bins[1:], hist_r, linewidth=1, color='r')
    ax_g.plot(bins[1:], hist_g, linewidth=1, color='g')
    ax_b.plot(bins[1:], hist_b, linewidth=1, color='b')

    ax_r.set_xlabel('Pixel Intensity')
    ax_r.set_ylabel('Pixel Frequency')
    ax_r.set_title('Red Channel')
    ax_r.set_xlim(0, 255)

    ax_g.set_xlabel('Pixel Intensity')
    ax_g.set_title('Green Channel')
    ax_g.set_xlim(0, 255)

    ax_b.set_xlabel('Pixel Intensity')
    ax_b.set_title('Blue Channel')
    ax_b.set_xlim(0, 255)

    from tikzplotlib import save as tikz_save
    tikz_save(f"hist_{ver}_1.tex")

    plt.show()


def plot_hist_combined(image, max_total, nbins=256):
    fig, ax = plt.subplots()
    hist, bins = np.histogram(image.ravel(), 256, [0, 256])
    hist_r, _ = np.histogram(image[0].ravel(), 256, [0, 256])
    hist_g, _ = np.histogram(image[1].ravel(), 256, [0, 256])
    hist_b, _ = np.histogram(image[2].ravel(), 256, [0, 256])

    ax.fill_between(bins[1:], 0, hist_r, facecolor='red', alpha=0.3, label="red")
    ax.fill_between(bins[1:], 0, hist_g, facecolor='green', alpha=0.3, label="green")
    ax.fill_between(bins[1:], 0, hist_b, facecolor='blue', alpha=0.3, label="blue")

    ax.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    ax.set_xlabel('Pixel intensity')
    ax.set_xlim(0, nbins)
    ax.legend()
    plt.show()


def plot_dpt_hist(image, nbins=256, ver=""):  # 65535):
    fig, ax = plt.subplots(figsize=(6, 3))
    m = image.max()
    hist, bins = np.histogram(image.ravel(), nbins, [0, m])
    hist = hist[1:]
    bins = bins[2:]
    print(hist)
    print(bins)
    ax.plot(bins, hist, 'gray', linewidth=2)
    ax.set_xlabel('Pixel Intensity')
    ax.set_ylabel('Pixel Frequency')
    ax.set_title('Depth Image')

    from tikzplotlib import save as tikz_save
    tikz_save(f"hist_dpt_{ver}_1.tex")

    plt.show()


def all_files():
    dist_r = []
    dist_b = []
    dist_g = []

    for filename in os.listdir("/datasets/elevator/preprocessed/rgb\\"):
        print(filename)
        recording = filename.rsplit("_", 1)[0]
        file = filename.rsplit("_", 1)[1]

        image1 = skimage.io.imread(
            f"../../datasets//elevator/preprocessed/rgb/{recording}_{file}")
        image2 = skimage.io.imread(f"../../datasets/elevator/{recording}/out/rgb/{recording}_{file}")

        dr, dg, db, m = calc_diff(image1, image2)
        dist_r.append(dr)
        dist_g.append(dg)
        dist_b.append(db)
    print("---")
    print(np.array(dist_r).mean())
    print(np.array(dist_g).mean())
    print(np.array(dist_b).mean())
    print(np.array(dist_r).std())
    print(np.array(dist_g).std())
    print(np.array(dist_b).std())


def all_sun_files():
    dist_r = []
    dist_b = []
    dist_g = []

    for filename in os.listdir("../../datasets/sun_rgbd/crop/image/test/"):
        print(filename)

        image1 = skimage.io.imread(
            f"../../datasets/sun_rgbd/crop/image/test/{filename}")
        image2 = skimage.io.imread(f"../../datasets/sun_rgbd/image/test/{filename}")

        dr, dg, db, m = calc_diff(image1, image2)
        dist_r.append(dr)
        dist_g.append(dg)
        dist_b.append(db)
    for filename in os.listdir("../../datasets/sun_rgbd/image/train/"):
        print(filename)

        image1 = skimage.io.imread(
            f"../../datasets/sun_rgbd/crop/image/train/{filename}")
        image2 = skimage.io.imread(f"../../datasets/sun_rgbd/image/train/{filename}")

        dr, dg, db, m = calc_diff(image1, image2)
        dist_r.append(dr)
        dist_g.append(dg)
        dist_b.append(db)
    print("---")
    print(np.array(dist_r).mean())
    print(np.array(dist_g).mean())
    print(np.array(dist_b).mean())
    print(np.array(dist_r).std())
    print(np.array(dist_g).std())
    print(np.array(dist_b).std())


def plot_dpt(image):
    fig, ax = plt.subplots(figsize=(8, 4))
    img = ax.imshow(image)
    fig.colorbar(img)
    ax.get_yaxis().set_visible(False)
    ax.get_xaxis().set_visible(False)
    plt.tight_layout()
    print(image.min())
    print(image.max())
    plt.show()


recording = "20190219_135155"
file = "_000094"


image1 = skimage.io.imread(
    f"{recording}{file}.jpg")
image2 = skimage.io.imread(f"{recording}{file}.jpg")
dpt2 = skimage.io.imread(
    f"{recording}{file}.png")
dpt1 = skimage.io.imread(
    f"{recording}{file}.png")
dpt1 = (dpt1 / 65535 * 255).astype(np.uint8)

plot_dpt_hist(dpt1, ver="after")
plot_dpt_hist(dpt2, ver="before")
plot_dpt(dpt2)
plot_dpt(dpt1)

dr, dg, db, m = calc_diff(image1, image2)

plot_hist_combined(image1, m)
plot_hist_combined(image2, m)
plot_hist_separate(image=image1, max_total=m, ver="after")
plot_hist_separate(image=image2, max_total=m, ver="before")

