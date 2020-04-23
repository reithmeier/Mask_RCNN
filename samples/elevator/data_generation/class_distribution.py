import argparse
import json
import os

ROOT_DIR = os.path.abspath("./../../..")


def main():
    class_cnt = {}
    labels_per_file = {}
    for filename in os.listdir(args.input):
        if filename.endswith(".json"):
            with open(args.input + "/" + filename) as in_file:
                data = json.load(in_file)
                annotations = data['completions'][0]["result"]
                for ann in annotations:
                    label = ann["value"]["polygonlabels"][0]
                    if label in class_cnt:
                        class_cnt[label] = class_cnt[label] + 1
                    else:
                        class_cnt[label] = 1
                label_cnt = len(annotations)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, help="Path of the input directory",
                        default=ROOT_DIR + "/datasets/elevator/preprocessed/labels/")
    args = parser.parse_args()
    main()
