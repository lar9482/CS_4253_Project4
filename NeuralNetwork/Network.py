from utils.Model import Model
import numpy as np
import random
import copy

class Network(Model):
    def __init__(self, input_layer_size, #Integer
                       deep_layer_sizes, #List of integers
                       output_layer_size, #Integer
                       activate = None,
                       d_activate = None):
        
        #The layer sizes for the neural network
        self.input_layer_size = input_layer_size
        self.deep_layer_sizes  = deep_layer_sizes
        self.output_layer_size = output_layer_size

        #The activation function
        self.activate = activate

        #Derivative of the activation function
        self.d_activate = d_activate

        #All of the layers in the neural network
        self.deep_layers = []

        #Raw values of the layers in the neural network
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
            else:
                self.weights.append(np.empty((deep_layer_sizes[i-1], deep_layer_sizes[i])))

        #Placing weights inbetween the last deep layer and the output layer
        self.weights.append(np.empty((deep_layer_sizes[len(deep_layer_sizes) - 1], output_layer_size)))

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
    
    

    #Implementation of 'Figure 1' from the instructions.
    def fit(self, X, Y):
        self.__initialize_weights()

        for example in range(0, len(X)):
            self.__feed_in_input(X[example])
            self.__feed_forward()

            actual_output = self.__get_actual_output(Y[example], self.output_layer_size)
            errors = self.__initialize_error_layers()

            
        


        