from utils.file_io import load_EMG_data, load_optdigits_data, load_spambase_data
from utils.shuffle import shuffle

from DecisionTree.DT import DT, Info_Gain
from NeuralNetwork.Network import Network
from NeuralNetwork.activation_functions import sigmoid, sigmoid_derivative
import numpy as np
from utils.N_Fold import N_Fold

def test_DT():
    (X, Y) = load_spambase_data(2000)
    (X, Y) = shuffle(X, Y)

    tree = DT(Info_Gain.Gini)
    (train, test) = N_Fold((X, Y), tree)
    print()

def main():
    (X, Y) = load_optdigits_data(100)
    (X, Y) = shuffle(X, Y)
    network = Network(len(X[0]), [20, 10], len(np.unique(Y)), sigmoid, sigmoid_derivative)
    network.fit(X, Y)
    
        
    # tree.fit(X, Y)
    # test = tree.predict(X)

if __name__ == "__main__":
    main()