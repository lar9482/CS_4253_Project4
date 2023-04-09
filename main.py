from utils.file_io import load_EMG_data, load_optdigits_data, load_spambase_data
from utils.shuffle import shuffle

from DecisionTree.DT import DT, Info_Gain
from NeuralNetwork.Network import Network
from NeuralNetwork.activation_functions import sigmoid, sigmoid_derivative
import numpy as np
from utils.N_Fold import N_Fold

from utils.graph import graph_DT_data

def test_DT():
    domains = [load_EMG_data]
    num_instances = [10, 11]
    info_gains = [Info_Gain.Entropy, Info_Gain.Gini]

    for domain in domains:
        data = {}
        for gain in info_gains:
            for instance in num_instances:
                (X, Y) = domain(instance)
                (X, Y) = shuffle(X, Y)
                tree = DT(gain)
                (train, test) = N_Fold((X, Y), tree)

                if (not gain in data.keys()):
                    data[gain] = [(train, test, instance)]
                else:
                    data[gain].append((train, test, instance))

        graph_DT_data(data, num_instances, domain)

def main():
    # (X, Y) = load_optdigits_data(100)
    # (X, Y) = shuffle(X, Y)
    # network = Network(len(X[0]), [100, 64, 10], len(np.unique(Y)), sigmoid, sigmoid_derivative, 0.5, 16, 25)
    # network.fit(X, Y)
    # accuracy = network.eval(X, Y)
    # print(accuracy)
    
        
    test_DT()

if __name__ == "__main__":
    main()