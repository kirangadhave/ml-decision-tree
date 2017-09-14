import numpy as np
import math
import data_extraction as de

def entropy(list_l):
    if list_l[0] == -1:
        list_l = np.delete(list_l, 0)
        
    value_count = [list_l.tolist().count(i) for i in set(list_l)]
    ent = 0
    for x in value_count:
       ent = ent - x/sum(value_count)*math.log(x/sum(value_count), 2)
    return(ent)
    
def information_gain(list_av, S, S_ent):
    unique_values = np.array([list_av[list_av[:,0] == x] for x in set(list_av[:,0])])
    ig = S_ent
    for x in unique_values:
        x_t = x[:,1]
        ig = ig - (x_t.size/S)*entropy(x_t)
    return ig
       
def highest_gain(features, label):
    S = len(label)
    S_ent = entropy(label)    
    gain = {}
    for i,x in enumerate(features.transpose()):
        g = information_gain(np.array([x, label]).transpose(), S, S_ent)
        gain[i] = g
    return max(gain, key = gain.get)