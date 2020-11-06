# **********************************************************************************************************************
#
# brief:    script to calculate mask files from label files
#
# author:   Lukas Reithmeier
# date:     18.05.2020
#
# **********************************************************************************************************************


import argparse
import json
import os

import numpy as np
import skimage.draw

ROOT_DIR = os.path.abspath("./../../..")

CLASS_NAMES_TO_ID = {
    "human_standing": 1,
    "human_sitting": 2,
    "human_lying": 3,
    "bag": 4,
    "box": 5,
    "crate": 6,
    "plant": 7,
    "chair": 8,
    "object": 9,
    "human_other": 10
}


def draw_polygon(polygon, width, height):
    y_values = np.array([p[0] * width / 100 for p in polygon])
    x_values = np.array([p[1] * height / 100 for p in polygon])
    rr, cc = skimage.draw.polygon(x_values, y_values)
    # labels might extend over boundaries, due to preprocessing
    rr = [height - 1 if r > height - 1 else r for r in rr]
    cc = [width - 1 if c > width - 1 else c for c in cc]
    rr = [0 if r < 0 else r for r in rr]
    cc = [0 if c < 0 else c for c in cc]
    return rr, cc


def filter_relations(relations, identifier):
    found = []
    for relation in relations:
        if relation["from_id"] == identifier:
            found.append(relation["to_id"])
        if relation["to_id"] == identifier:
            found.append(relation["from_id"])
    return found


def find_result(results, identifier):
    for result in results:
        if result["id"] == identifier:
            return result["value"]["points"]
    return []


def calculate_mask(lbl_file_name):
    with open(args.input + lbl_file_name) as lbl_file:
        labels = json.load(lbl_file)
    results = labels["completions"][-1]["result"]  # always get the latest entry
    if len(results) == 0:  # no labels
        return np.zeros([0, 0, 0], dtype=np.bool), np.array([])

    height = results[0]["original_height"]
    width = results[0]["original_width"]
    instance_count = len(results)

    mask = np.zeros([height, width, instance_count], np.bool)
    class_ids = np.zeros([instance_count], np.int)

    relations = []
    # relations indicate that two polygons belong to the same object instance in the image
    # therefore relations need to be found and masks need to be merged
    for result in results:
        if result["type"] != "relation":
            continue
        # keys: from_id, to_id
        relations.append(result)

    skip_ids = []  # skip already merged polygons

    i = 0
    for result in results:
        # skip non-polygons
        if result["type"] != "polygonlabels":
            continue
        if result["id"] in skip_ids:
            continue

        rr, cc = draw_polygon(result["value"]["points"], width, height)

        mask[rr, cc, i] = True

        # find relation
        relations_for_id = filter_relations(relations, result["id"])
        for relation in relations_for_id:
            rel_rr, rel_cc = draw_polygon(find_result(results, relation), width, height)
            # merge with related polygon
            mask[rel_rr, rel_cc, i] = True
            # don't use merged polygon again
            skip_ids.append(relation)

        label_txt = result["value"]["polygonlabels"][0]
        class_ids[i] = CLASS_NAMES_TO_ID[label_txt]
        i = i + 1

    return mask, class_ids


def process(lbl_file_name):
    mask, class_ids = calculate_mask(lbl_file_name=lbl_file_name)
    np.save(args.output + lbl_file_name + ".mask.npy", mask)
    np.save(args.output + lbl_file_name + ".class_ids.npy", class_ids)


def main():
    for filename in os.listdir(args.input):
        if os.path.isfile(args.input + filename):
            process(lbl_file_name=filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", type=str, help="Path of the output directory",
                        default=ROOT_DIR + "/datasets/elevator/preprocessed/labels/")
    parser.add_argument("-i", "--input", type=str, help="Path of the input directory",
                        default=ROOT_DIR + "/datasets/elevator/preprocessed/labels/")

    args = parser.parse_args()
    main()
