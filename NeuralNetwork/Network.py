from utils.Model import Model
import numpy as np
import random
import os

class Network(Model):
    def __init__(self, input_layer_size, #Integer
                       deep_layer_sizes, #List of integers
                       output_layer_size, #Integer
                       activate = None, #Function
                       d_activate = None, #Function
                       alpha = 0.5,
                       batch_size = 16,
                       epochs = 100):
        
        #The layer sizes for the neural network
        self.input_layer_size = input_layer_size
        self.deep_layer_sizes  = deep_layer_sizes
        self.output_layer_size = output_layer_size

        #The learning rate
        self.alpha = alpha

        #The batch size for minibatch training
        self.batch_size = batch_size

        #The number of iterations to train the network on 
        self.epochs = epochs

        #The activation function
        self.activate = activate

        #Derivative of the activation function
        self.d_activate = d_activate

        #All of the layers in the neural network(after the activation)
        self.deep_layers = []

        #Raw values of the layers in the neural network(before the activation)
        self.raw_deep_layers = []

        #The input layer
        self.deep_layers.append(np.empty((input_layer_size, 1)))
        self.raw_deep_layers.append(np.empty((input_layer_size, 1)))

        #The deep layers
        for deep_layer_size in deep_layer_sizes:
            self.deep_layers.append(np.empty((deep_layer_size, 1)))
            self.raw_deep_layers.append(np.empty((deep_layer_size, 1)))

        #The output layer
        self.deep_layers.append(np.empty((output_layer_size, 1)))
        self.raw_deep_layers.append(np.empty((output_layer_size, 1)))

        #The weights between each layer
        self.weights = []
        for i in range(0, len(deep_layer_sizes)):

            #Placing weights inbetween the input layer and the 1st deep layer
            if (i == 0):
                self.weights.append(np.empty((input_layer_size, deep_layer_sizes[i])))

            #Placing weights inbetween the last seen layer, and the current layer
            else:
                self.weights.append(np.empty((deep_layer_sizes[i-1], deep_layer_sizes[i])))

        #Placing weights inbetween the last deep layer and the output layer
        self.weights.append(np.empty(
                                (deep_layer_sizes[len(deep_layer_sizes) - 1], output_layer_size)
                           ))

    def __initialize_weights(self):
        for weight_layer in range(0, len(self.weights)):
            for i in range(0, len(self.weights[weight_layer])):
                for j in range(0, len(self.weights[weight_layer][0])):
                    self.weights[weight_layer][i, j] = random.uniform(-1, 1)

    def __feed_in_input(self, X):
        for i in range(0, len(X)):
            self.deep_layers[0][i] = X[i]
            self.raw_deep_layers[0][i] = X[i] 

    def __feed_forward(self):

        #Processing data through the all of the inner deep layers and to the output layer
        for layer in range(1, len(self.deep_layers)):
            
            #For each node in the current layer.
            for j in range(0, len(self.deep_layers[layer])):

                #Basically perform a matrix multiplication with the weights and the previous layer
                in_j = 0
                for i in range(0, len(self.deep_layers[layer-1])):
                    in_j += self.weights[layer-1][i][j]*self.deep_layers[layer-1][i]
                a_j = self.activate(in_j)

                self.deep_layers[layer][j] = a_j
                self.raw_deep_layers[layer][j] = in_j
    
    def __get_actual_output(self, Y, num_classes):
        actual_output = np.empty((num_classes, 1))
        
        #Y is assigned an index that matches its class.
        #It will be assigned one, while the rest of the elements are assigned zero
        for class_name in range(0, num_classes):
            if (int(Y[0]) == class_name):
                actual_output[class_name] = 1
            else:
                actual_output[class_name] = 0
        
        return actual_output
    
    def __initialize_error_layers(self):
        errors = []
        for i in range(0, len(self.deep_layers)):
            errors.append(np.copy(self.deep_layers[i]))

        return errors
    
    def __backpropagation(self, actual_output):
        delta_errors = self.__initialize_error_layers()

        #For deep layer in the neural network
        for layer in range(len(self.deep_layers)-1, -1, -1):
            
            #For the output layer, calculate the error vector
            #between the actual output vector and the predicted one. 
            if (layer == len(self.deep_layers)-1):

                #For every node in the output layer, compute the error with respect to the target outputs
                #and begin feeding them back through the network.
                for j in range(0, len(self.deep_layers[layer])):

                    delta_errors[layer][j] = self.d_activate(
                                            self.raw_deep_layers[layer][j]
                                       ) * (actual_output[j] - self.deep_layers[layer][j])
            
            #For the rest of the layers, propagate the error vectors back
            else:
                for i in range(0, len(self.deep_layers[layer])):
                    weighted_error_j = 0

                    for j in range(0, len(self.deep_layers[layer+1])):
                        weighted_error_j += self.weights[layer][i][j]*delta_errors[layer+1][j]
                    
                    delta_errors[layer][i] = self.d_activate(
                                            self.raw_deep_layers[layer+1][j]
                                       ) * (weighted_error_j)
            
        return delta_errors
    
    def __update_weights(self, delta_errors):

        #Scanning through every weight layer to update them.
        for weight_layer in range(0, len(self.weights)):
            
            layer = self.weights.index(self.weights[weight_layer])
            for i in range(0, len(self.weights[weight_layer])):
                for j in range(0, len(self.weights[weight_layer][0])):
                    
                    self.weights[weight_layer][i][j] = self.weights[weight_layer][i][j] + (self.alpha)*(self.deep_layers[layer][i])*(delta_errors[layer+1][j])

    #Implementation of 'Figure 1' from the instructions.
    def fit(self, X, Y):
        self.__initialize_weights()

        for epoch in range(0, self.epochs):
            for example in range(0, len(X)):
                self.__feed_in_input(X[example])
                self.__feed_forward()
            
                actual_output = self.__get_actual_output(Y[example], self.output_layer_size)
                delta_errors = self.__backpropagation(actual_output)

                self.__update_weights(delta_errors)
            
            print('Process %s processing epoch %s' % (str(str(os.getpid())), str(epoch+1)))

    def predict(self, X):
        prediction = np.empty((len(X), 1))

        for example in range(0, len(X)):
            self.__feed_in_input(X[example])
            self.__feed_forward()
            raw_prediction = self.deep_layers[len(self.deep_layers) - 1]
            
            prediction[example] = np.argmax(raw_prediction)
        
        return prediction
    
    def eval(self, X, Y):
        actual_Y = self.predict(X)
        correct_guesses = 0
        for class_id in range(0, len(Y)): 
            if (Y[class_id][0] == actual_Y[class_id][0]):
                correct_guesses += 1

        return correct_guesses / len(Y)     