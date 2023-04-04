from DecisionTree.DT_node import DT_node

class DT_branch(DT_node):
    def __init__(self, attribute, value):
        self.value = value
        self.child_node = None
        
        super().__init(attribute)

    def __lt__(self, other):
        return self.value <= other.value