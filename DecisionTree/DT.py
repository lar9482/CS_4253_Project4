from utils.Model import Model
from enum import Enum
from operator import attrgetter

from DecisionTree.DT_node import DT_node
from DecisionTree.DT_leaf_node import DT_leaf_node
from DecisionTree.DT_branch import DT_branch

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
        self.tree = self.learn_tree((X, Y),                                     #Examples
                                    [attr for attr in range(0, len(X[0]))],     #Attributes
                                    None)                                       #Parent_examples
    
    #Implementation of figure 19.5, which is the decision tree learning algorithm.
    def learn_tree(self, examples, attributes, parent_examples):
        print(len(examples[0]))

        #If the examples(X vector/Y vector) are empty
        if (len(examples[0]) == 0 or len(examples[1]) == 0):
            return DT_leaf_node(self.plurality_value(parent_examples))

        #If the Y vector passed in all have the same classifiction
        elif len(np.unique(examples[1])) == 1:
            return DT_leaf_node(examples[1][0])
        
        #If there are no more attributes to process
        elif len(attributes) == 0:
            return DT_leaf_node(self.plurality_value(examples))

        else:
            A = self.importance(examples, attributes)
            new_tree = DT_node(A)
            attribute_values = np.unique(examples[0][:, A])

            for value in attribute_values:
                #Getting the indices(or rows in the examples), where the selected attribute matches the current attribute value test.
                subset_examples_indices = np.where(examples[0][:, A] == value)

                #Use the indices to get the X/Y vectors that match 'attr_value' at the specified 'attribute'.
                subset_X = [examples[0][indice, :] for indice in subset_examples_indices][0]
                subset_Y = [examples[1][indice] for indice in subset_examples_indices][0]

                #Removing the selected attribute 'A'
                subset_attributes = attributes.copy()
                subset_attributes.remove(A)

                #Recursively build the subtree
                sub_tree = self.learn_tree((subset_X, subset_Y), subset_attributes, examples)

                #Add a branch to 'new_tree' with label (A = v) and the 'sub_tree'
                new_branch = DT_branch(A, value)
                new_branch.child_node = sub_tree
                new_tree.branches.add(new_branch)
            
            return new_tree
    
    #Gets the most common classification among the examples
    def plurality_value(self, examples):
        unique, indices = np.unique(examples[1], return_inverse=True)
        return unique[np.argmax(np.bincount(indices))]
        
    def importance(self, examples, attributes):

        #Keeping track of the attribute that maximizes the information that's gained
        argmax_attribute = attributes[0]
        max_info_gain = -sys.maxsize-1

        #Given all the current attributes, find the one that maximizes info_gain
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

        #Calculating the remainer
        #(I.e sum of node impurities for every possible split at the inputted attribute)
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

            #Get number of examples classified under 'class_name'
            points_belonging_to_class = len(np.where(examples[1] == class_name)[0])

            #Getting the number of examples
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
        
    def predict(self, X):
        prediction = np.empty((len(X), 1))
        
        for i in range(0, len(prediction)):

            #Starting at the root of the tree
            curr_node = self.tree

            #Traverse down to a leaf node
            while (not curr_node.__class__.__name__ == DT_leaf_node.__name__):

                #Testing every branch in the current node until its not necessary
                for branch in curr_node.branches:

                    #Once the tracked attribute value matches the attribute value in X,
                    #Go down the tree
                    if (X[i][curr_node.attribute] <= branch.value):
                        curr_node = branch.child_node
                        break
            
            #Once a leaf node has been found, log it in 'prediction'
            prediction[i, 0] = curr_node.classification[0]
        
        return prediction
        
                    

