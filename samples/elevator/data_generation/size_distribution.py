# **********************************************************************************************************************
#
# brief:    script to print the distribution of different sizes in the data set
#
# author:   Lukas Reithmeier
# date:     23.04.2020
#
# **********************************************************************************************************************


import argparse
import json
import os
import numpy as np
from matplotlib import pyplot as plt

ROOT_DIR = os.path.abspath("./../../..")




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, help="Path of the input directory",
                        default="")
    args = parser.parse_args()
    main()