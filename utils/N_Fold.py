import numpy as np
from utils.graph import graph_training_testing_NN_data

def N_Fold(examples, model, n = 5):
    iterations = 0
    #Difference measures the roughly equal partition size of the dataset
    difference = int((len(examples[0]) - (len(examples[0]) % n)) / n)
    startIndex = 0
    endIndex = difference

    accu_train = 0
    accu_test = 0

    #For all possible blocks 
    while (endIndex < len(examples[0])):
        #Dividing the dataset from the calculated difference
        train_X = np.vstack((examples[0][0:startIndex:1, :], examples[0][endIndex:len(examples[0]):1, :]))
        train_Y = np.vstack((examples[1][0:startIndex:1, :], examples[1][endIndex:len(examples[0]):1, :]))

        test_X = examples[0][startIndex:endIndex:1, :]
        test_Y = examples[1][startIndex:endIndex:1, :]

        #Fitting the current model
        model.fit(train_X, train_Y)

        #Evalulating both the training and testing dataset
        model_train_accu = model.eval(train_X, train_Y)
        model_test_accu = model.eval(test_X, test_Y)
        
        #Adding the calculating accuracies to the total accuracies
        accu_train += model_train_accu
        accu_test += model_test_accu

        startIndex = endIndex
        endIndex += difference
        iterations += 1
        print(startIndex)
        print("Training Accuracy: %s" % (model_train_accu))
        print("Testing Accuracy: %s" % (model_test_accu))
    
    #Return the average training/testing accuracy.
    return ((accu_train / (iterations)), (accu_test / (iterations)))

#Basically the same implementation, but it collects more data for the neural network.
def N_Fold_NN(examples, model, domain_name, n = 5):
    iterations = 0
    #Difference measures the roughly equal partition size of the dataset
    difference = int((len(examples[0]) - (len(examples[0]) % n)) / n)
    startIndex = 0
    endIndex = difference

    accu_train = 0
    accu_test = 0

    all_training_acc = []
    all_testing_acc = []

    #For all possible blocks 
    while (endIndex < len(examples[0])):
        #Dividing the dataset from the calculated difference
        train_X = np.vstack((examples[0][0:startIndex:1, :], examples[0][endIndex:len(examples[0]):1, :]))
        train_Y = np.vstack((examples[1][0:startIndex:1, :], examples[1][endIndex:len(examples[0]):1, :]))

        test_X = examples[0][startIndex:endIndex:1, :]
        test_Y = examples[1][startIndex:endIndex:1, :]

        #Fitting the current model
        (curr_training_acc, curr_testing_acc) = model.fit(train_X, train_Y, test_X, test_Y)

        #Appending the tracked accuracies
        for i in range(0, len(curr_training_acc)):
            if (i == len(all_training_acc)):
                all_training_acc.append(curr_training_acc[i] / n)
            else:
                all_training_acc[i] += curr_training_acc[i] / n

            if (i == len(all_testing_acc)):
                all_testing_acc.append(curr_testing_acc[i] / n)
            else:
                all_testing_acc[i] += curr_testing_acc[i] / n

        #Evalulating both the training and testing dataset
        model_train_accu = model.eval(train_X, train_Y)
        model_test_accu = model.eval(test_X, test_Y)
        
        #Adding the calculating accuracies to the total accuracies
        accu_train += model_train_accu
        accu_test += model_test_accu

        startIndex = endIndex
        endIndex += difference
        iterations += 1
        print(startIndex)
        print("Training Accuracy: %s" % (model_train_accu))
        print("Testing Accuracy: %s" % (model_test_accu))
    
    #Graph the initial results
    graph_training_testing_NN_data(domain_name, 
                                   all_training_acc, 
                                   all_testing_acc, 
                                   str(model.deep_layer_sizes), 
                                   str(model.alpha),
                                   str(model.decay))

    #Return the average training/testing accuracy.
    return ((accu_train / (iterations)), (accu_test / (iterations)))