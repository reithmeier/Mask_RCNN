# **********************************************************************************************************************
#
# brief:    simple script to rename label files
#
# author:   Lukas Reithmeier
# date:     23.04.2020
#
# **********************************************************************************************************************


import argparse
import json
import os

ROOT_DIR = os.path.abspath("./../../..")


def main():
    if not os.path.exists(args.output):
        os.mkdir(args.output)

    for filename in os.listdir(args.input):
        if filename.endswith(".json"):
            with open(args.input + "/" + filename) as in_file:
                data = json.load(in_file)
                new_file_name = data['task_path'].split('/')[-1].replace('jpg', 'json')
                print(new_file_name)
                with open(args.output + "/" + new_file_name, 'w') as out_file:
                    json.dump(data, out_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", type=str, help="Path to save the labels",
                        default=ROOT_DIR + "/datasets/elevator/out/labels")
    parser.add_argument("-i", "--input", type=str, help="Path to load the labels from",
                        default=ROOT_DIR +"/datasets/elevator/out/completions")
    args = parser.parse_args()
    main()
