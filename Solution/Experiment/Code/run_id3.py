import os
import numpy as np
import data_extraction as de
import decision_tree_calculations as dtc

training_file = os.path.abspath( "../Dataset/updated_train.txt")
data = np.array(de.get_data(training_file))

### Build Features

# Gets features described above and maps the label to 1 = '+' and 0 = '-'
data = de.extract_features_and_labels(data)
features = data[:,:-1]
labels = data[:,-1]
### End Features