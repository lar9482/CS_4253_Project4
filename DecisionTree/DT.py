from utils.Model import Model
from enum import Enum

from DecisionTree.DT_node import DT_node

import numpy as np
import sys

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
            A = self.importance(examples, attributes)

    def importance(self, examples, attributes):
        argmax_attribute = attributes[0]
        max_info_gain = -sys.maxsize-1

        for attribute in attributes:
            curr_info_gain = self.information_gain(examples, attribute)
            if (curr_info_gain > max_info_gain):
                max_info_gain = curr_info_gain
                argmax_attribute = attribute
    
        return argmax_attribute

    def information_gain(self, examples, attribute):

        #Get all possible attribute values from the 'examples' dataset for the given attribute.
        attribute_values = np.unique(examples[0][:, attribute])

        #Impurity of the inputted examples
        parent_impurity = self.impurity(examples)

        remainder = 0
        for attr_value in attribute_values:

            #Getting the indices(or rows in the examples), where the selected attribute matches the current attribute value test.
            subset_examples_indices = np.where(examples[0][:, attribute] == attr_value)

            #Use the indices to get the X/Y vectors that match 'attr_value' at the specified 'attribute'.
            subset_examples_X = [examples[0][indice, :] for indice in subset_examples_indices][0]
            subset_examples_Y = [examples[1][indice] for indice in subset_examples_indices][0]

            remainder += (len(subset_examples_X)/len(examples))*self.impurity((subset_examples_X, subset_examples_Y))
        
        return parent_impurity - remainder

    def impurity(self, examples):

        #Extracting all possible class types from the examples
        all_classes = np.unique(examples[1])

        #Keep track of the fractional points associated with each class
        all_point_class_fracts = []

        #Initially calculate all fractions of points that belong to each class
        for class_name in all_classes:
            points_belonging_to_class = len(np.where(examples[1] == class_name)[0])
            all_points = len(examples[1])

            point_class_fract = (points_belonging_to_class / all_points)
            all_point_class_fracts.append(point_class_fract)
        
        #SOURCE: Slide 16 from 'SupervisedLearningDecisionTree.pptx'

        #For misclassification, get the maximum point_class fraction and subtract it from 1
        if (self.info_gain == Info_Gain.Mis_classify):
            max_pointclass_fracs = max(all_point_class_fracts)
            return 1 - max_pointclass_fracs
        
        #For Entropy, plug all point_class fractions into entropy function 
        elif (self.info_gain == Info_Gain.Entropy):
            sum = 0
            for point_class_fract in all_point_class_fracts:
                sum += (point_class_fract) * np.log2(point_class_fract)

            return -sum
        
        #For Gini Index, plug all point_class fractions into Gini_Index function 
        elif (self.info_gain == Info_Gain.Gini):
            sum = 0
            for point_class_fract in all_point_class_fracts:
                sum += (point_class_fract) * (1 - point_class_fract)

            return sum

        else:
            raise Exception('Impurity: Invalid information gain request made!')


    def plurality_value(self, Y):
        pass