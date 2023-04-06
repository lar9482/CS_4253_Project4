from abc import ABC, abstractmethod

class Model:
    
    @abstractmethod
    def fit(self, X, Y):
        pass
    
    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def eval(self, X, Y):
        pass