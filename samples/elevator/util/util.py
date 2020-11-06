# **********************************************************************************************************************
#
# brief:    utility functions for dataset classes
#
# author:   Lukas Reithmeier
# date:     12.05.2020
#
# **********************************************************************************************************************


import json

import numpy as np
import skimage.draw

if __name__ == "__main__":
    # sometimes a rare bug occurs, don't know why
    # g_c = np.array([203, 249, 239])
    # g_r = np.array([146, 165, 117])
    # skimage.draw.polygon(g_r, g_c)
    g_r = np.array([1, 2, 8])
    g_c = np.array([1, 7, 4])
    g_rr, g_cc = skimage.draw.polygon(g_r, g_c)

    # leads to:
    # TypeError: int() argument must be a string, a bytes-like object or a number, not '_NoValueType'


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


def create_mask(lbl_full_path, class_name_to_id):
    """
    Convert polygons to a bitmap mask of shape
    [height, width, instance_count]

    :param lbl_full_path:
    :param class_name_to_id:
    :return:
    """

    with open(lbl_full_path) as lbl_file:
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
        class_ids[i] = class_name_to_id[label_txt]
        i = i + 1

    return mask, class_ids
