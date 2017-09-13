import os
import numpy as np
import operator
import decision_tree_calculations as dtc
import data_extraction as de
import matplotlib.pyplot as plt

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

depth_tree = 0
tree_feature_order = []

def create_tree(data, labels, depth=0):
    global depth_tree
    global tree_order
    
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
    
    
    if(depth_tree < depth or len(tree_feature_order) == 0):
        depth_tree = depth
        tree_feature_order.append(best_feature_label)
    
    for val in unique_vals:
        sub_labels = labels[:]
        tree[best_feature_label][val] = create_tree(split_data(data.tolist(), best_feature, val), sub_labels, depth+1)
    return tree

def iterate_on_tree(tree, x):
    if type(tree) is np.int64:
        return tree
    key_no = list(tree.keys())[0]
    val = x[key_no]
    temp = x[:key_no]
    temp = np.append(temp, x[key_no:]) 
    return iterate_on_tree(tree[key_no][val], temp)
    
def predict(file):
    prediction = []
    test_data = de.extract_features_and_labels(np.array(de.get_data(file)))
    actual_output = test_data[:,-1]
    test_data = test_data[:,:-1]
    for x in test_data:
        val = iterate_on_tree(tree, x)
        prediction.append(val)
    correct = 0
    for i,x in enumerate(prediction):
        if x == actual_output[i]:
            correct += 1
    return prediction

def predict_and_calc_accuracy(file, actual_output):
    preds = predict(file)
    correct = 0
    for i,x in enumerate(preds):
        if x == actual_output[i]:
            correct += 1
    return float(correct/len(actual_output)*100)

def prune_tree():
    print("")    
    


training_file = os.path.abspath( "../Dataset/updated_train.txt")
test_file = os.path.abspath( "../Dataset/updated_test.txt")
print("")
print("")
print("Reading Training Data: ")
print("Reading Training Data: Done")
data = np.array(de.get_data(training_file))
print("")
print("Extracting Features from training data:")
print("Extracting Features from training data: Done")
data = de.extract_features_and_labels(data)
labels = de.create_labels(data)
print("")

print("Running ID3")
print("Tree created")
tree = create_tree(data, labels)
print("")
print("")
print(tree)
print("")
print("")

print("Reading and processing test_data")
test_data = np.array(de.get_data(test_file))
test_data =  de.extract_features_and_labels(test_data)
print("Classifying Test Data Results")
print("Accuracy is : " + str(predict_and_calc_accuracy(test_file, test_data[:,-1])))

actual_output = test_data[:,-1]
predictions = predict(test_file)
plt.scatter(range(len(actual_output)),actual_output, marker = 'o')
plt.scatter(range(len(actual_output)),actual_output, marker = 'x')
plt.show()
print(tree_order)