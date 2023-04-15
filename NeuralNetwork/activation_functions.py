import math

def sigmoid(x):
    if (x > 50):
        return 0.99999
    elif (x < -50):
        return 0.00001
    else:
        return (math.exp(x)) / (1 + math.exp(x))

def sigmoid_derivative(x):
    return sigmoid(x)*(1-sigmoid(x))


def tanh(x):
    if (x > 50):
        return 0.99999
    elif (x < -50):
        return -0.99999
    else:
        return (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))
    

def tanh_derivative(x):
    return 1 - tanh(x)*tanh(x)