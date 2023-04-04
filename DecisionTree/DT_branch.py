from DecisionTree.DT_node import DT_node

class DT_Branch(DT_node):
    def __init__(self, attribute, value):
        
        self.attribute_value = value
        super().__init(attribute)


