from abc import ABC, abstractmethod

class Model:
    
    @abstractmethod
    def fit(X, Y):
        pass
    
    @abstractmethod
    def predict(X):
        pass

    @abstractmethod
    def eval(X, Y):
        pass