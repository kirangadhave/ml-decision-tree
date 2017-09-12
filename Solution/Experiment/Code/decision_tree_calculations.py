import numpy as np
import math

def entropy(list_l):
    value_count = [list_l.tolist().count(i) for i in set(list_l)]
    ent = 0
    for x in value_count:
       ent = ent - x/sum(value_count)*math.log(x/sum(value_count), 2)
    return(ent)
    
def information_gain(list_av):
    unique_values = np.array([list_av[list_av[:,0] == x] for x in set(list_av[:,0])])
    print(unique_values)
       