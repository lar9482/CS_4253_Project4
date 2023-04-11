from utils.file_io import load_EMG_data, load_optdigits_data, load_spambase_data
from utils.shuffle import shuffle

from DecisionTree.DT import DT, Info_Gain
from NeuralNetwork.Network import Network
from NeuralNetwork.activation_functions import sigmoid, sigmoid_derivative
import numpy as np
from utils.N_Fold import N_Fold

from utils.graph import graph_DT_data

def test_DT():
    domains = [load_spambase_data]
    num_instances = [10, 50, 100, 500, 1000, 2500, 4500]
    # num_instances = [4500]
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

def test_NN():
    epochs = 100
    instances = 1000
    (X, Y) = load_optdigits_data(100)
    (X, Y) = shuffle(X, Y)

    node_options = [[len(X[0]), len(np.unique(Y))], 
                    [2*len(X[0]), int((len(X[0]) + len(np.unique(Y))) / 2), int(len(np.unique(Y))/2)]
                    [int(len(X[0])/2), int((len(X[0]) + len(np.unique(Y))) / 2), int(2*len(np.unique(Y)))], 
                    [len(X[0])],
                    [len(np.unique(Y))]]
    
    learning_rate = [0.01, 0.1, 0.5, 1]
def main():
    # (X, Y) = load_spambase_data(100)
    # (X, Y) = shuffle(X, Y)
    # network = Network(len(X[0]), [64], len(np.unique(Y)), sigmoid, sigmoid_derivative, 0.5, 16, 1)
    # network.fit(X, Y)
    # accuracy = network.eval(X, Y)
    # print(accuracy)
    
        
    test_DT()

if __name__ == "__main__":
    main()