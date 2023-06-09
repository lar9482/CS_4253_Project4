from utils.file_io import load_EMG_data, load_optdigits_data, load_artificial_dataset, save_NN_data
from utils.N_Fold import N_Fold, N_Fold_NN
from utils.graph import graph_DT_data
from utils.shuffle import shuffle

from DecisionTree.DT import DT, Info_Gain
from NeuralNetwork.Network import Network
from NeuralNetwork.activation_functions import sigmoid, sigmoid_derivative

import numpy as np
from sklearn.preprocessing import normalize
from multiprocessing import Process, Manager


def test_DT():
    # domains = [load_EMG_data, load_optdigits_data, load_spambase_data]
    num_instances = [50, 100, 500, 1000, 2500, 3500, 4500]
    domains = [load_EMG_data, load_optdigits_data]
    # num_instances = [50, 100, 200, 400, 600, 800, 1000]
    info_gains = [Info_Gain.Entropy, Info_Gain.Gini]

    for domain in domains:
        data = {}
        for gain in info_gains:
            for instance in num_instances:
                (X, Y) = domain(instance)
                (X, Y) = shuffle(X, Y)
                if (domain.__name__ != 'load_artificial_dataset'):
                    X = normalize(X)
                tree = DT(gain)
                (train, test) = N_Fold((X, Y), tree)

                if (not gain in data.keys()):
                    data[gain] = [(train, test, instance)]
                else:
                    data[gain].append((train, test, instance))

        graph_DT_data(data, num_instances, domain)

def run_NN(name, alpha, decay, node_option, domain, instances, epochs,
           node_option_data,
           lock):
    
    (X, Y) = domain(instances)
    (X, Y) = shuffle(X, Y)

    if (domain.__name__ != 'load_artificial_dataset'):
        X = normalize(X)
    
    network = Network(len(X[0]), node_option, len(np.unique(Y)),
                      sigmoid,
                      sigmoid_derivative,
                      alpha,
                      decay,
                      epochs)
    
    (train_acc, test_acc) = N_Fold_NN((X, Y), network, name)

    lock.acquire()
    node_option_data.append({
                             'node_options': str(node_option), 
                             'alpha': str(alpha),
                             'decay': str(decay),
                             'training_accuracy': str(train_acc),
                             'testing_accuracy': str(test_acc)})
    lock.release()


def test_NN():

    epochs = 100
    instances = 100

    domains = [load_artificial_dataset, load_optdigits_data, load_EMG_data]
    for domain in domains:

        (X, Y) = domain(instances)
        (X, Y) = shuffle(X, Y)

        if (domain.__name__ != 'load_artificial_dataset'):
            X = normalize(X)


        node_options = [[5*int(len(X[0]))]]
        
        learning_rates = [0.1, 0.25, 0.5, 0.75]
        decay_rates = [0.0001]

        
        with Manager() as manager:
            all_processes = []
            lock = manager.Lock()
            node_option_data = manager.list()

            for alpha in learning_rates:
                for decay in decay_rates:
                    for node_option in node_options:
                        process = Process(target=run_NN, args=(
                            domain.__name__,
                            alpha,
                            decay,
                            node_option,
                            domain,
                            instances,
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

            data_list = list(node_option_data)

            save_NN_data(data_list, domain.__name__)

            
def main():
    test_DT()
    # test_NN()

if __name__ == "__main__":
    main()