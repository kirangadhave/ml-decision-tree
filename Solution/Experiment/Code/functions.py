import numpy as np

def get_data(file):
    data = []
    with open(file, 'r') as f:
        for x in f:
            data.append(x.strip().split(' ', 1))
    return data

def extract_features_and_labels(data):
    # First Name longer than last name
    flln_feature = []
    for x in data[:,1]:
        flln_feature.append(int(x.split(' ')[0] >= x.split(' ')[-1]))
    flln_feature = np.array(flln_feature)
    
    # Middle Name present
    midd_feature = []
    for x in data[:,1]:
        midd_feature.append(int(len(x.split(' ')) > 2))
    midd_feature = np.array(midd_feature)
    
    
    # First name start and end letters are same
    first_se_feature = []
    for x in data[:,1]:
        first_se_feature.append(int(str.lower(x.split(' ')[0][0]) == str.lower(x.split(' ')[0][-1])))
    first_se_feature = np.array(first_se_feature)
    
    # Alphabetical names
    alpha_feature = []
    for x in data[:,1]:
        alpha_feature.append(int(str.lower(x.split(' ')[0][0]) <= str.lower(x.split(' ')[0][-1])))
    alpha_feature = np.array(alpha_feature)
    
    # Is second letter vowel?
    sec_vow_feature = []
    for x in data[:,1]:
        sec_vow_feature.append(int(str.lower(x.split(' ')[0][1]) in ['a', 'e', 'i', 'o', 'u']))
    sec_vow_feature = np.array(sec_vow_feature)
    
    # Is first letter vowel?
    first_vow_feature = []
    for x in data[:,1]:
        first_vow_feature.append(int(str.lower(x.split(' ')[0][0]) in ['a', 'e', 'i', 'o', 'u']))
    first_vow_feature = np.array(first_vow_feature)
    
    # Number of letters in last name even?
    last_even_features = []
    for x in data[:,1]:
        last_even_features.append(int(len(x.split(' ')[-1])%2 == 0))
    last_even_features = np.array(last_even_features)
    
    # Number of letters in first name even?
    first_even_features = []
    for x in data[:,1]:
        first_even_features.append(int(len(x.split(' ')[0])%2 == 0))
    first_even_features = np.array(first_even_features)
    
    label = []
    for x in data[:,0]:
        if x == '+':
            label.append(1)
        else:
            label.append(0)
    
    label = np.array(label)

    return np.array([flln_feature, midd_feature, first_se_feature, alpha_feature, sec_vow_feature, first_vow_feature, last_even_features, first_even_features, label]).transpose()
