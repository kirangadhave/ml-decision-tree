import numpy as np

def get_data(file):
    data = []
    with open(file, 'r') as f:
        for x in f:
            data.append(x.strip().split(' ', 1))
    return data

def extract_features_and_labels(data):
    # 0 First Name longer than last name
    flln_feature = []
    for x in data[:,1]:
        flln_feature.append(int(len(x.split(' ')[0]) > len(x.split(' ')[-1])))
    flln_feature = np.array(flln_feature)
    
    # 1 Middle Name present
    midd_feature = []
    for x in data[:,1]:
        midd_feature.append(int(len(x.split(' ')) > 2))
    midd_feature = np.array(midd_feature)
    
    
    # 2 First name start and end letters are same
    first_se_feature = []
    for x in data[:,1]:
        first_se_feature.append(int(str.lower(x.split(' ')[0][0]) == str.lower(x.split(' ')[0][-1])))
    first_se_feature = np.array(first_se_feature)
    
    # 3 Alphabetical names
    alpha_feature = []
    for x in data[:,1]:
        alpha_feature.append(int(str.lower(x.split(' ')[0][0]) <= str.lower(x.split(' ')[0][-1])))
    alpha_feature = np.array(alpha_feature)
    
    # 4 Is second letter vowel?
    sec_vow_feature = []
    for x in data[:,1]:
        if len(x.split(' ')[0]) > 1:
            sec_vow_feature.append(int(str.lower(x.split(' ')[0][1]) in ['a', 'e', 'i', 'o', 'u']))
        else:
            sec_vow_feature.append(0)
    sec_vow_feature = np.array(sec_vow_feature)
    
    # 5 Is first letter vowel?
    first_vow_feature = []
    for x in data[:,1]:
        first_vow_feature.append(int(str.lower(x.split(' ')[0][0]) in ['a', 'e', 'i', 'o', 'u']))
    first_vow_feature = np.array(first_vow_feature)
    
    # 6 Number of letters in last name even?
    last_even_features = []
    for x in data[:,1]:
        last_even_features.append(int(len(x.split(' ')[-1])%2 == 0))
    last_even_features = np.array(last_even_features)
    
    # 7 Number of letters in first name even?
    first_even_features = []
    for x in data[:,1]:
        first_even_features.append(int(len(x.split(' ')[0])%2 == 0))
    first_even_features = np.array(first_even_features)
    
    # Map labels
    label = []
    for x in data[:,0]:
        if x == '+':
            label.append(1)
        else:
            label.append(0)
    
    label = np.array(label)
    
    
    return np.array([flln_feature, midd_feature, first_se_feature, alpha_feature, sec_vow_feature, first_vow_feature, last_even_features, first_even_features, label]).transpose()

#    return np.array([flln_feature, midd_feature, first_se_feature, alpha_feature, sec_vow_feature, first_vow_feature, last_even_features, first_even_features, label]).transpose()

    
def add_header(data):
    headers = list(range(0, data.shape[1] - 1))
    headers.append(-1)
    headers = np.array(headers)
    data = data.transpose()
    headers = headers.T
    return np.insert(data, 0, headers,1).transpose()

def create_labels(data):
    headers = list(range(0, data.shape[1] - 1))
    return headers

def remove_header(data, axis=0):
    print(np.delete(data, 0, axis))
    return np.delete(data, 0, axis)

    























