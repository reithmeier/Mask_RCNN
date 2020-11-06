# **********************************************************************************************************************
#
# brief:    simple script to add labels to index file
#
# author:   Lukas Reithmeier
# date:     23.04.2020
#
# **********************************************************************************************************************


import argparse


def main():
    with open(args.input) as in_file:
        with open(args.output, 'w') as out_file:
            for line in in_file:
                print(line)
                lbl_file = line.split(" ")[-1].replace("depth_intrinsics", "labels")
                new_line = line.replace("\n", "") + " " + lbl_file
                out_file.write(new_line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", type=str, help="output index file", default="./out/new_all.txt")
    parser.add_argument("-i", "--input", type=str, help="input index file", default="./out/all.txt")
    args = parser.parse_args()
    main()
