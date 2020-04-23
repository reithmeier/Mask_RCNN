import argparse
import json
import os

ROOT_DIR = os.path.abspath("./../../..")


def main():
    all_filename = args.input + "all.txt"
    trimmed_filename = args.input + "trimmed.txt"
    with open(trimmed_filename, "w") as out_file:
        with open(all_filename) as in_file:
            for line in in_file:
                lbl_filename = args.input + line.split(' ')[4][2:].strip()
                print(lbl_filename)
                with open(lbl_filename) as lbl_file:
                    data = json.load(lbl_file)
                    annotations = data['completions'][0]["result"]
                    if len(annotations) > 0:
                        out_file.write(line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, help="Path of the input directory",
                        default=ROOT_DIR + "/datasets/elevator/out/")
    args = parser.parse_args()
    main()
