from DecisionTree.DT_node import DT_node

class DT_leaf_node(DT_node):
    def __init__(self, classification):
        self.classification = classification

        