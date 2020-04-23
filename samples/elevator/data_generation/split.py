import argparse
import os

import numpy as np

ROOT_DIR = os.path.abspath("./../../..")

TRN = 0.6
VAL = 0.2
TST = 0.2


def main():
    if not os.path.exists(args.output):
        os.mkdir(args.output)

    num_img = sum(1 for line in open(args.input))

    idx_trn = round(TRN * num_img)
    idx_val = idx_trn + round(VAL * num_img)
    idx_tst = idx_val + round(TST * num_img)

    print(num_img)
    print(idx_trn)
    print(idx_val)
    print(idx_tst)

    perm = np.random.permutation(range(0, num_img))
    print(perm)

    trn_indices = perm[0:idx_trn]
    val_indices = perm[idx_trn:idx_val]
    tst_indices = perm[idx_val:idx_tst]

    print(trn_indices)
    print(val_indices)
    print(tst_indices)

    i = 0
    with open(args.output + "/train.txt", "w") as trn_file:
        with open(args.output + "/validation.txt", "w") as val_file:
            with open(args.output + "/test.txt", "w") as tst_file:
                with open(args.input) as in_file:
                    for line in in_file:
                        if i in trn_indices:
                            trn_file.write(line)
                        if i in val_indices:
                            val_file.write(line)
                        if i in tst_indices:
                            tst_file.write(line)

                        i = i + 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", type=str, help="Path to save the split files",
                        default=ROOT_DIR + "/datasets/elevator/out/split")
    parser.add_argument("-i", "--input", type=str, help="Input index file",
                        default=ROOT_DIR + "/datasets/elevator/out/all.txt")
    args = parser.parse_args()
    main()
