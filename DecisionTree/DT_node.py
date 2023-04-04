from sortedcontainers import SortedList

class DT_node:
    def __init__(self, attribute):

        #The attribute that's being tested.
        self.attribute = attribute

        #References to branches.
        self.branches = SortedList()