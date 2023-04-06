from utils.Model import Model
import numpy as np

class Network(Model):
    def __init__(self, input_layer_size, #Integer
                       deep_layer_sizes, #List of integers
                       output_layer_size, #Integer
                       activate = None):
        
        #The layer sizes for the neural network
        self.input_layer_size = input_layer_size
        self.deep_layer_sizes  = deep_layer_sizes
        self.output_layer_size = output_layer_size

        #The activation function
        self.activate = activate

        #The input layer
        self.input_layer = np.empty((input_layer_size, 1))

        #The output layer
        self.output_layer = np.empty((output_layer_size, 1))

        #The deep layers
        self.deep_layers = []
        for deep_layer_size in deep_layer_sizes:
            self.deep_layers.append(np.empty((deep_layer_size, 1)))

        #The weights
        self.weights = []
        for i in range(0, len(deep_layer_sizes)):

            #Placing weights inbetween the input layer and the 1st deep layer
            if (i == 0):
                self.weights.append(np.empty((input_layer_size, deep_layer_sizes[i])))
            else:
                self.weights.append(np.empty((deep_layer_sizes[i-1], deep_layer_sizes[i])))

        #Placing weights inbetween the last deep layer and the output layer
        self.weights.append(np.empty((deep_layer_sizes[len(deep_layer_sizes) - 1], output_layer_size)))
        


        