import numpy as np

NUM_IMG = 5050

NUM_TRN = int(0.70 * NUM_IMG)
NUM_VAL = int(0.20 * NUM_IMG)
NUM_TST = int(0.10 * NUM_IMG)

if __name__ == '__main__':
    print(NUM_TRN)
    print(NUM_VAL)
    print(NUM_TST)

    perm = np.random.permutation(range(1, NUM_IMG + 1))
    print(perm)

    # training
    for i in range(0, NUM_TRN):
        print(str(i) + " " + "img13labels-00{:04}".format(perm[i]))
    print("---")
    # validation
    for i in range(NUM_TRN, NUM_TRN + NUM_VAL):
        print(str(i) + " " + "img13labels-00{:04}".format(perm[i]))
    print("---")
    # test
    for i in range(NUM_TRN + NUM_VAL, NUM_TRN + NUM_VAL + NUM_TST):
        print(str(i) + " " + "img13labels-00{:04}".format(perm[i]))
