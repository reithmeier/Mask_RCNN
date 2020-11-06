# **********************************************************************************************************************
#
# brief:    read image data from ros bag files and persist it
#
# author:   Lukas Reithmeier
# date:     23.04.2020
#
# **********************************************************************************************************************

import argparse
import os
import time

import cv2
import numpy as np
import pyrealsense2 as rs

ROOT_DIR = os.path.abspath("./../../..")


def strip(file):
    """
    removes args.directory from file
    :return: file without args.directory
    """
    return file.replace(args.directory, '.')


def write_intrinsics(intrinsics, file):
    """
    writes intrinsics in json format to a file
    """
    f = open(file, "w")
    f.write("{\n")
    f.write("\t\"width\":" + str(intrinsics.width) + ",\n")
    f.write("\t\"height\":" + str(intrinsics.height) + ",\n")
    f.write("\t\"ppx\":" + str(intrinsics.ppx) + ",\n")
    f.write("\t\"ppy\":" + str(intrinsics.ppy) + ",\n")
    f.write("\t\"fx\":" + str(intrinsics.fx) + ",\n")
    f.write("\t\"fy\":" + str(intrinsics.fy) + ",\n")
    f.write("\t\"model\":" + str(int(intrinsics.model)) + ",\n")
    f.write("\t\"coeffs\":" + str(intrinsics.coeffs) + "\n")
    f.write("}")
    f.close()


def process_frames(rgb_dir, dpt_dir, dpt_raw_dir, rgb_itc_dir, dpt_itc_dir, lbl_dir):
    """
    reads color and depth frame
    aligns the frames to the color frame
    writes depth frame, color frame and intrinsics to files
    :param lbl_dir: directory to save label file to
    :param dpt_raw_dir: directory to save unaligned depth frames to
    :param rgb_dir: directory to save color frames to
    :param dpt_dir: directory to save depth frames to
    :param rgb_itc_dir: directory to save color intrinsics to
    :param dpt_itc_dir: directory to save depth intrinsics to
    """

    prefix = args.bag_file.split(".")[0]  # name of file without extension
    index_file = open(args.directory + "/" + "all.txt", "a")

    try:
        config = rs.config()
        print(args.bag_directory + args.bag_file)
        rs.config.enable_device_from_file(config, args.bag_directory + args.bag_file, repeat_playback=False)
        pipeline = rs.pipeline()
        config.enable_stream(rs.stream.depth)
        config.enable_stream(rs.stream.color)
        profile = pipeline.start(config)
        align_to = rs.stream.color
        align = rs.align(align_to)

        rgb_profile = profile.get_stream(rs.stream.color)
        dpt_profile = profile.get_stream(rs.stream.depth)
        rgb_intrinsics = rgb_profile.as_video_stream_profile().get_intrinsics()
        dpt_intrinsics = dpt_profile.as_video_stream_profile().get_intrinsics()

        i = 0
        while True:
            frames = pipeline.wait_for_frames()
            dpt_raw_frame = frames.get_depth_frame()
            aligned_frames = align.process(frames)

            bgr_frame = aligned_frames.get_color_frame()
            dpt_frame = aligned_frames.get_depth_frame()

            bgr_image = np.asanyarray(bgr_frame.get_data())
            rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_RGB2BGR)
            dpt_image = np.asanyarray(dpt_frame.get_data())
            dpt_raw_image = np.asanyarray(dpt_raw_frame.get_data())

            print("Saving frame:", i)

            rgb_file = rgb_dir + "/" + prefix + "_" + str(i).zfill(6) + ".jpg"
            dpt_file = dpt_dir + "/" + prefix + "_" + str(i).zfill(6) + ".png"
            dpt_raw_file = dpt_raw_dir + "/" + prefix + "_" + str(i).zfill(6) + ".png"
            rgb_itc_file = rgb_itc_dir + "/" + prefix + "_" + str(i).zfill(6) + ".json"
            dpt_itc_file = dpt_itc_dir + "/" + prefix + "_" + str(i).zfill(6) + ".json"
            lbl_file = lbl_dir + "/" + prefix + "_" + str(i).zfill(6) + ".json"
            cv2.imwrite(rgb_file, rgb_image)
            cv2.imwrite(dpt_file, dpt_image)
            cv2.imwrite(dpt_raw_file, dpt_raw_image)

            write_intrinsics(rgb_intrinsics, rgb_itc_file)
            write_intrinsics(dpt_intrinsics, dpt_itc_file)
            index_file.write(strip(rgb_file) + " " +
                             strip(dpt_file) + " " +
                             strip(rgb_itc_file) + " " +
                             strip(dpt_itc_file) + " " +
                             strip(lbl_file) + " " +
                             strip(dpt_raw_file) + "\n")

            time.sleep(1.000)  # dont use all frames
            i += 1
    finally:
        index_file.close()
        pass


def main():
    if not os.path.exists(args.directory):
        os.mkdir(args.directory)
    rgb_dir = args.directory + "/rgb"
    dpt_dir = args.directory + "/depth"
    dpt_raw_dir = args.directory + "/depth_raw"
    rgb_itc_dir = args.directory + "/rgb_intrinsics"
    dpt_itc_dir = args.directory + "/depth_intrinsics"
    lbl_dir = args.directory + "/labels"

    if not os.path.exists(rgb_dir):
        os.mkdir(rgb_dir)
    if not os.path.exists(dpt_raw_dir):
        os.mkdir(dpt_raw_dir)
    if not os.path.exists(dpt_dir):
        os.mkdir(dpt_dir)
    if not os.path.exists(rgb_itc_dir):
        os.mkdir(rgb_itc_dir)
    if not os.path.exists(dpt_itc_dir):
        os.mkdir(dpt_itc_dir)
    if not os.path.exists(lbl_dir):
        os.mkdir(lbl_dir)

    process_frames(rgb_dir, dpt_dir, dpt_raw_dir, rgb_itc_dir, dpt_itc_dir, lbl_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--directory", type=str, help="Path to save the images",
                        default=ROOT_DIR + "/datasets/elevator/Intel_N_6-3_3-3_2/out/")
    parser.add_argument("-i", "--bag_directory", type=str, help="Bag file directory",
                        default=ROOT_DIR + "/datasets/elevator/Intel_N_6-3_3-3_2/")
    parser.add_argument("-b", "--bag_file", type=str, help="Bag file to read",
                        default="Intel_N_6-3_3-3_2.bag")
    args = parser.parse_args()

    main()
