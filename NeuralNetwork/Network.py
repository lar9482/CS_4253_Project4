from utils.Model import Model
import numpy as np
import random

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

        #The input layer
        self.deep_layers.append(np.empty((input_layer_size, 1)))

        #The deep layers
        for deep_layer_size in deep_layer_sizes:
            self.deep_layers.append(np.empty((deep_layer_size, 1)))

        #The output layers
        self.deep_layers.append(np.empty((output_layer_size, 1)))

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
                    
    def _feed_in_input(self, X):
        X.transpose()
        pass

    def fit(self, X, Y):
        self.__initialize_weights()

        # for example in range(0, len(X)):

        # print()
        


        