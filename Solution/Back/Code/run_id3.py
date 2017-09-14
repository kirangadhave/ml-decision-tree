import os
import numpy as np
import operator
import decision_tree_calculations as dtc
import data_extraction as de
import matplotlib.pyplot as plt
import sys


training_file = os.path.abspath(sys.argv[1])
test_file = os.path.abspath(sys.argv[2])

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
    
    
    try:
        return iterate_on_tree(tree[key_no][val], temp)
    except:
        if(val == 0):
            val = 1
        if(val==1):
            val = 0
        return iterate_on_tree(tree[key_no][val], temp)
    
def predict(tree, file):
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

def predict_and_calc_accuracy(tree, file, actual_output):
    preds = predict(tree, file)
    correct = 0
    for i,x in enumerate(preds):
        if x == actual_output[i]:
            correct += 1
    return float(correct/len(actual_output)*100)

def create_pruned_tree(data, labels, depth=0, depth_limit = 1):
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
 
    if(depth_limit == 0):
        return most_freq(classes)    
        
    if(depth_tree < depth or len(tree_feature_order) == 0):
        depth_tree = depth
        tree_feature_order.append(best_feature_label)
    
    for val in unique_vals:
        sub_labels = labels[:]
        tree[best_feature_label][val] = create_pruned_tree(split_data(data.tolist(), best_feature, val), sub_labels, depth+1, depth_limit - 1)
    return tree


def question1c():
    data = np.array(de.get_data(training_file))
    data = de.extract_features_and_labels(data)
    labels = de.create_labels(data)

    tree = create_tree(data, labels)

    test_data = np.array(de.get_data(test_file))
    test_data =  de.extract_features_and_labels(test_data)

    actual_output = test_data[:,-1]
    predictions = predict(tree, test_file)
        
    count = 0
    for i,x in enumerate(actual_output):
        if x == predictions[i]:
            count += 1
    
    accuracy = count*100/len(actual_output)
    output = []
    output.append("Question 1-c")
    output.append("########################################")
    output.append("Accuracy on training data: " + str(round(accuracy,2)))
    output.append("Depth of the tree is: " + str(depth_tree))
    output.append("########################################")
    with open(os.path.abspath("TraceFiles/question1c.trace"), 'w', ) as file:
        file.write('\n'.join(output))
    for x in output:
        print(x)

def question1d():
    data = np.array(de.get_data(training_file))
    data = de.extract_features_and_labels(data)
    labels = de.create_labels(data)

    tree = create_tree(data, labels)

    test_data = np.array(de.get_data(test_file))
    test_data =  de.extract_features_and_labels(test_data)

    actual_output = test_data[:,-1]
    predictions = predict(tree, test_file)
        
    count = 0
    for i,x in enumerate(actual_output):
        if x == predictions[i]:
            count += 1
    
    accuracy = count*100/len(actual_output)
    output = []
    output.append("Question 1-d")
    output.append("########################################")
    output.append("Accuracy on test data: " + str(round(accuracy,2)))
    output.append("Depth of the tree is: " + str(depth_tree))
    output.append("########################################")
    with open(os.path.abspath("TraceFiles/question1d.trace"), 'w', ) as file:
        file.write('\n'.join(output))
    for x in output:
        print(x)

def question2():
    split_folder = os.path.abspath(sys.argv[3])
    depths = list(map(int, sys.argv[4:]))
    
    kfold_files = []
    for (dirpath, d, files) in os.walk(split_folder):
        kfold_files = [os.path.join(dirpath,x) for x in files]
    
    data_0 = np.array(de.get_data(kfold_files[0]))
    data_1 = np.array(de.get_data(kfold_files[1]))
    data_2 = np.array(de.get_data(kfold_files[2]))
    data_3 = np.array(de.get_data(kfold_files[3]))
    
    train_1 = np.concatenate([data_0, data_1, data_2])
    test_data_1 =  de.extract_features_and_labels(data_3)
    op1 = test_data_1[:, -1]
    
    train_2 = np.concatenate([data_1, data_2, data_3])
    test_data_2 =  de.extract_features_and_labels(data_0)
    op2 = test_data_2[:, -1]
    
    train_3 = np.concatenate([data_2, data_3, data_0])
    test_data_3 =  de.extract_features_and_labels(data_1)
    op3 = test_data_3[:, -1]
    
    train_4 = np.concatenate([data_3, data_0, data_1])
    test_data_4 =  de.extract_features_and_labels(data_2)
    op4 = test_data_4[:, -1]
    
    depth_accuracy_dict = {}
    std_dev_dict = {}
    
    
    for depth in depths:
        train_d1 = de.extract_features_and_labels(train_1)
        labels_1 = de.create_labels(train_d1)
        
        train_d2 = de.extract_features_and_labels(train_2)
        labels_2 = de.create_labels(train_d2)
        
        train_d3 = de.extract_features_and_labels(train_3)
        labels_3 = de.create_labels(train_d3)
        
        train_d4 = de.extract_features_and_labels(train_4)
        labels_4 = de.create_labels(train_d4)
        
        tree_1 = create_pruned_tree(train_d1, labels_1, 0, depth)
        depth_tree = 0
        tree_2 = create_pruned_tree(train_d2, labels_2, 0, depth)        
        depth_tree = 0
        tree_3 = create_pruned_tree(train_d3, labels_3, 0, depth)               
        depth_tree = 0
        tree_4 = create_pruned_tree(train_d4, labels_4, 0, depth)
        depth_tree = 0
        
        
        p1 = predict_and_calc_accuracy(tree_1, kfold_files[3], op1)
        p2 = predict_and_calc_accuracy(tree_2, kfold_files[0], op2)
        p3 = predict_and_calc_accuracy(tree_3, kfold_files[1], op3)
        p4 = predict_and_calc_accuracy(tree_4, kfold_files[2], op4)
        
        depth_accuracy_dict[depth] = round(sum([p1, p2, p3, p4])/4,2)
        std_dev_dict[depth] = round(np.std([p1,p2,p3,p4]),2)
        
    max_acc = max(depth_accuracy_dict, key = depth_accuracy_dict.get)
    output = []
    
    for x in depth_accuracy_dict:
        output.append("============")
        output.append("Depth = " + str(x))
        output.append("Average Accuracy = " + str(depth_accuracy_dict[x]))
        output.append("Standard Deviation = " + str(std_dev_dict[x]))
        output.append("============")
    
    output.append("Max accuracy " + str(depth_accuracy_dict[max_acc]) + " is found at depth " + str(max_acc))
    output.append("Optimal depth = " + str(max_acc))
    output.append("============")
    
    data = np.array(de.get_data(training_file))
    data = de.extract_features_and_labels(data)
    labels = de.create_labels(data)
    tree = create_pruned_tree(data, labels, 0, max_acc)

    test_data = np.array(de.get_data(test_file))
    test_data =  de.extract_features_and_labels(test_data)

    actual_output = test_data[:,-1]
    predictions = predict(tree, test_file)
        
    count = 0
    for i,x in enumerate(actual_output):
        if x == predictions[i]:
            count += 1
    
    accuracy = round(count*100/len(actual_output),2)
    
    output.append("Training using depth = " + str(max_acc))
    output.append("Accuracy = " + str(accuracy))
    
    
    with open(os.path.abspath("TraceFiles/question2.trace"), 'w', ) as file:
        file.write('\n'.join(output))
    for x in output:
        print(x)

if(len(sys.argv) > 3):
    question2()
elif(sys.argv[1] == sys.argv[2]):
    question1c()
elif(sys.argv[1] != sys.argv[2]):
    question1d()