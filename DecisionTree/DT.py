from utils.Model import Model
from enum import Enum

from DecisionTree.DT_node import DT_node

import numpy as np

class Info_Gain(Enum):
    Mis_classify = 0
    Entropy = 1
    Gini = 2

class DT(Model):
    def __init__(self, info_gain = Info_Gain.Mis_classify):
        self.info_gain = info_gain
        self.tree = None

    def fit(self, X, Y):
        self.tree = self.learn_tree((X, Y), 
                                    [attr for attr in range(0, len(X[0]))], 
                                    None,
                                    self.tree)
    
    def learn_tree(self, examples, attributes, parent_examples, curr_branch):
        
        #If X/Y are empty
        if (len(examples[0]) == 0 or len(examples[1]) == 0):
            print('Plurality parent examples')

        #If Y avaliable have the same classifiction
        elif len(np.unique(examples[1])) == 1:
            print('Empty current examples')
        
        #If there are no more attributes to process
        elif len(attributes) == 0:
            print('Plurality current examples')

        else:
            self.importance(examples, attributes)

    def importance(self, examples, attributes):
        for attribute in attributes:
            pass
        pass

    def plurality_value(self, Y):
        pass
