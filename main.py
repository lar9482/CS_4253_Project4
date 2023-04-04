from utils.file_io import load_EMG_data, load_optdigits_data, load_spambase_data
from utils.shuffle import shuffle
from DecisionTree.DT import DT, Info_Gain
import numpy as np
def main():
    (X, Y) = load_spambase_data(5000)
    (X, Y) = shuffle(X, Y)

    tree = DT(Info_Gain.Gini)
    tree.fit(X, Y)
    # test = tree.impurity((X, Y))
    print()


if __name__ == "__main__":
    main()