from utils.file_io import load_EMG_data, load_optdigits_data, load_spambase_data
from utils.shuffle import shuffle
from DecisionTree.DT import DT
import numpy as np
def main():
    (X, Y) = load_spambase_data(4500)
    (X, Y) = shuffle(X, Y)

    tree = DT()
    tree.fit(X, Y)


if __name__ == "__main__":
    main()