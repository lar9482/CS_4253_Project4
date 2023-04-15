from utils.file_io import load_EMG_data, load_optdigits_data, load_spambase_data
from utils.shuffle import shuffle

from DecisionTree.DT import DT, Info_Gain
from NeuralNetwork.Network import Network
from NeuralNetwork.activation_functions import sigmoid, sigmoid_derivative, tanh, tanh_derivative
import numpy as np
from utils.N_Fold import N_Fold, N_Fold_NN
from sklearn.preprocessing import normalize

from multiprocessing import Process, Manager
from utils.graph import graph_DT_data, graph_NN_data

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
           node_option_data,
           lock):
    
    network = Network(len(X[0]), node_option, len(np.unique(Y)),
                      sigmoid,
                      sigmoid_derivative,
                      alpha,
                      1,
                      epochs)
    
    (train_acc, test_acc) = N_Fold((X, Y), network)

    lock.acquire()
    print(alpha)
    if (not (str(node_option) + str(alpha)) in node_option_data.keys()):
        node_option_data[str(node_option) + str(alpha)] = [(train_acc, test_acc)]
    else:
        
        node_option_data[str(node_option) + str(alpha)].append((train_acc, test_acc))
    lock.release()


def test_NN():

    epochs = 100
    instances = 500
    domains = [load_EMG_data]
    for domain in domains:

        (X, Y) = domain(instances)
        (X, Y) = shuffle(X, Y)
        X = normalize(X)
        node_options = [[int(len(X[0])), int(len(np.unique(Y)))], 
                        [int(2*len(X[0])), int((len(X[0]) + len(np.unique(Y))) / 2), int(len(np.unique(Y))/2)],
                        [int((len(X[0]) + len(np.unique(Y))) / 2)]]
        learning_rates = [0.01, 0.5, 1]

        decay_rate = 0.0001

        
        with Manager() as manager:
            all_processes = []
            lock = manager.Lock()
            node_option_data = manager.dict()

            for alpha in learning_rates:
                for node_option in node_options:
                    process = Process(target=run_NN, args=(
                        alpha,
                        node_option,
                        X,
                        Y,
                        epochs,
                        node_option_data,
                        lock
                    ))
                    all_processes.append(process)

            #Start all of the subprocesses
            for process in all_processes:
                process.start()

            #Wait for all subprocesses to finish before continuing
            for process in all_processes:
                process.join()

            data_dict = dict(node_option_data)

            graph_NN_data(data_dict, domain)

            
def main():
    (X, Y) = load_optdigits_data(10)
    (X, Y) = shuffle(X, Y)
    network = Network(len(X[0]), [60], len(np.unique(Y)), sigmoid, sigmoid_derivative, 0.4, 0.0001, 5)

    X = normalize(X)

    N_Fold_NN((X, Y), network, load_optdigits_data.__name__)
    # network.fit(X, Y)
    # accuracy = network.eval(X, Y)
    # print(accuracy)
    
        
    # test_DT()
    # test_NN()

if __name__ == "__main__":
    main()