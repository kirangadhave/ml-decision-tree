import os
import math
import numpy as np
import functions as fn

training_file = os.path.abspath( "../Dataset/updated_train.txt")
data = np.array(fn.get_data(training_file))

### Build Features

# Gets features described above and maps the label to 1 = '+' and 0 = '-'
features = fn.extract_features_and_labels(data)


### End Features