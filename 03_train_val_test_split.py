# This script splits data into train, val, and test sets
# Modify paths below prior to running.
# To run: python3 03_train_val_test_split.py

import pandas as pd
from sklearn.model_selection import train_test_split

# modify these paths
'''
data_path = '/data/anau/SOD_classification/data/4_classes/propagated_1_1/'
data_file = '/data/anau/SOD_classification/data/4_classes/propagated_1_1/stages.csv.20221114_correct.4_classes.prop_1_1'

data_path = '/data/anau/SOD_classification/data/4_classes/'
data_file = '/data/anau/SOD_classification/data/4_classes/stages.csv.20221114_correct.4_classes'

data_path = '/data/anau/SOD_classification/data/3_classes/propagated_0_0/'
data_file = '/data/anau/SOD_classification/data/3_classes/propagated_0_0/stages.csv.20230201.3_classes.multiple.propagated.merged.plus_head'

data_path = '/data/anau/SOD_classification/data/4_classes/propagated_0_0/'
data_file = '/data/anau/SOD_classification/data/4_classes/propagated_0_0/stages.csv.20230201.4_classes.multiple.propagated.merged.plus_head'

data_path = '/data/anau/SOD_classification/data/4_classes/'
data_file = '/data/anau/SOD_classification/data/4_classes/stages.csv.20230201.annotated.multiple.4_classes.plus_head'

data_path = '/data/anau/SOD_classification/data/4_classes/null_hypoth/'
data_file = '/data/anau/SOD_classification/data/4_classes/null_hypoth/20230208.4_classes.propagated_k_10'
'''

data_df = pd.read_csv(data_file, header=None)

# perform train-val-test split 70%-15%-15% split
X_train, X_test = train_test_split(data_df, test_size=0.3, random_state=1, shuffle=True)
X_val, X_test = train_test_split(X_test, test_size=0.5, random_state=1, shuffle=True)

print(data_df.shape, X_train.shape, X_val.shape, X_test.shape)

### modify dataset names if needed ###
X_train.to_csv(data_path+'train_70', index=False, header=False)
X_val.to_csv(data_path+'val_15', index=False, header=False)
X_test.to_csv(data_path+'test_15', index=False, header=False)
