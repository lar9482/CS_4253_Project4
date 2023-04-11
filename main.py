from utils.file_io import load_EMG_data, load_optdigits_data, load_spambase_data
from utils.shuffle import shuffle

from DecisionTree.DT import DT, Info_Gain
from NeuralNetwork.Network import Network
from NeuralNetwork.activation_functions import sigmoid, sigmoid_derivative
import numpy as np
from utils.N_Fold import N_Fold
from sklearn.preprocessing import normalize

from multiprocessing import Process, Manager
from utils.graph import graph_DT_data

def test_DT():
    domains = [load_EMG_data, load_optdigits_data, load_spambase_data]
    num_instances = [50, 100, 500, 1000, 2500, 3500, 4500]
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

def run_NN(alpha, node_option, X, Y, epochs,
           alpha_accuracy,
           lock):
    
    network = Network(len(X[0]), node_option, len(np.unique(Y)),
                      sigmoid,
                      sigmoid_derivative,
                      alpha,
                      1,
                      epochs)
    
    (train_acc, test_acc) = N_Fold((X, Y), network)

    lock.acquire()
    if (not str(alpha) in alpha_accuracy.keys()):
        alpha_accuracy[alpha] = [(train_acc, test_acc, str(node_option))]
    else:
        alpha_accuracy[alpha].append((train_acc, test_acc, str(node_option)))
    lock.release()


def test_NN():

    epochs = 50
    instances = 200
    domains = [load_EMG_data, load_optdigits_data, load_spambase_data]
    for domain in domains:

        (X, Y) = domain(instances)
        (X, Y) = shuffle(X, Y)
        X = normalize(X)
        node_options = [[int(len(X[0])), int(len(np.unique(Y)))], 
                        [int(2*len(X[0])), int((len(X[0]) + len(np.unique(Y))) / 2), int(len(np.unique(Y))/2)],
                        [int(len(X[0])/2), int((len(X[0]) + len(np.unique(Y))) / 2), int(2*len(np.unique(Y))) ]]
        learning_rates = [0.1, 0.5, 1]
        print(str(node_options[0]))
        with Manager() as manager:
            all_processes = []
            lock = manager.Lock()
            alpha_accuracy = manager.dict()

            for alpha in learning_rates:
                for node_option in node_options:
                    process = Process(target=run_NN, args=(
                        alpha,
                        node_option,
                        X,
                        Y,
                        epochs,
                        alpha_accuracy,
                        lock
                    ))
                    all_processes.append(process)

            
def main():
    # (X, Y) = load_optdigits_data(100)
    # (X, Y) = shuffle(X, Y)
    # network = Network(len(X[0]), [128, 10], len(np.unique(Y)), sigmoid, sigmoid_derivative, 0.5, 16, 50)

    # X = normalize(X)
    # network.fit(X, Y)
    # accuracy = network.eval(X, Y)
    # print(accuracy)
    
        
    # test_DT()
    test_NN()

if __name__ == "__main__":
    main()