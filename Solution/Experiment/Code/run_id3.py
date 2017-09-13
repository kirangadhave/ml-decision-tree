import os
import numpy as np
import operator
import decision_tree_calculations as dtc
import data_extraction as de

training_file = os.path.abspath( "../Dataset/updated_train.txt")
data = np.array(de.get_data(training_file))

data = de.extract_features_and_labels(data)
labels = de.create_labels(data)

def split_data(data, split_column, value):
    newdata = []
    for feature in data:
        if feature[split_column] == value:
            feat_red = feature[:split_column]
            feat_red.extend(feature[split_column+1:])
            newdata.append(feat_red)
    return np.array(newdata)

def most_freq(classes):
    class_count = {}
    for x in classes:
        if x not in class_count.keys():
            class_count[x] = 0
        class_count[x] += 1
    sorted_counts = sorted(class_count.items(), key = operator.itemgetter(1), reverse = True)
    return sorted_counts[0][0]

def create_tree(data, labels): 
    classes = [x[-1] for x in data]
    if classes.count(classes[0]) == len(classes):
        return classes[0]
    if len(data[0]) == 1:
        return most_freq(classes)
    best_feature = dtc.highest_gain(data[:,:-1], data[:,-1])
    best_feature_label = labels[best_feature]
    tree = {best_feature_label : {}}
   
    del(labels[best_feature])
    feature_values = [x[best_feature] for x in data]
    unique_vals = set(feature_values)
    
    for val in unique_vals:
        sub_labels = labels[:]
        tree[best_feature_label][val] = create_tree(split_data(data.tolist(), best_feature, val), sub_labels)
    return tree

tree = create_tree(data, labels)
print(tree)