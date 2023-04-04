class DT_node:
    def __init__(self, attribute_index, attribute_value):
        self.attribute_index = attribute_index
        self.attribute_value = attribute_value

        self.output = -1
        
        self.branches = []