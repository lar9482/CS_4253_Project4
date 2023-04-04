class DT_node:
    def __init__(self, attribute):

        #The attribute that's being tested.
        self.attribute = attribute

        #References to child nodes.
        self.child_nodes = []