import numpy as np

def shuffle(X, Y):
    if len(X) != len(Y):
        raise Exception('Shuffle: Incompatible dataset')
    
    indices = np.arange(len(X))
    np.random.shuffle(indices)

    X = X[indices]
    Y = Y[indices]
    
    return(X, Y)