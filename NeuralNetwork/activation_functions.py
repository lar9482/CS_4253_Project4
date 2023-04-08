import math

def sigmoid(x):
    if (x > 50):
        return 0.999
    elif (x < -50):
        return -0.999
    else:
        return (math.exp(x)) / (1 + math.exp(x))

def sigmoid_derivative(x):
    return sigmoid(x)*(1-sigmoid(x))