import numpy as np

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