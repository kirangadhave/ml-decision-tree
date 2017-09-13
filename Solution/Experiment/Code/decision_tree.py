class DT_Node:
    
    def __init__(self, featureNo, true_value, root = False):
        self.is_root = root
        self.featureNo = featureNo
        self.true_value = true_value
        self.children = []
    