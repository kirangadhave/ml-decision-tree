import decision_tree as dt

def generate_tree(data):
    features = data[:,:-1]
    labels = data[:,-1]
    root = dt.DT_Node(features, labels)
    
    
    
    
    
    return root