from utils.Model import Model
from enum import Enum

from DecisionTree.DT_node import DT_node

import numpy as np
import math

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

    def impurity(self, examples):
        all_classes = np.unique(examples[1])
        all_point_class_fracts = []

        #Initially calculate all fractions of points that belong to each class
        for class_name in all_classes:
            points_belonging_to_class = len(np.where(examples[1] == class_name)[0])
            all_points = len(examples[1])

            point_class_fract = (points_belonging_to_class / all_points)
            all_point_class_fracts.append(point_class_fract)
        
        #SOURCE: Slide 16 from 'SupervisedLearningDecisionTree.pptx'
        if (self.info_gain == Info_Gain.Mis_classify):
            max_pointclass_fracs = max(all_point_class_fracts)
            return 1 - max_pointclass_fracs
        
        elif (self.info_gain == Info_Gain.Entropy):
            sum = 0
            for point_class_fract in all_point_class_fracts:
                sum += (point_class_fract) * np.log2(point_class_fract)

            return -sum
        
        elif (self.info_gain == Info_Gain.Gini):
            sum = 0
            for point_class_fract in all_point_class_fracts:
                sum += (point_class_fract) * (1 - point_class_fract)

            return sum

        else:
            raise Exception('Impurity: Invalid information gain request made!')


    def plurality_value(self, Y):
        pass
