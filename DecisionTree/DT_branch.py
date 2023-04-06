from DecisionTree.DT_node import DT_node

class DT_branch(DT_node):
    def __init__(self, attribute, value):
        self.value = value
        self.child_node = None
        
        super().__init__(attribute)

    #Less than 'operator'
    #Allows for 'DT_nodes' to keep a sorted list of 'DT_branches' based on the labelled value
    def __lt__(self, other):
        return self.value <= other.value