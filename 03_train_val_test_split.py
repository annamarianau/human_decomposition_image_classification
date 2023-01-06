# This script split data into train, val, and test sets
import pandas as pd
from sklearn.model_selection import train_test_split

# modify these paths 
data_path = '/data/anau/SOD_classification/data/3_classes/propagated_0_0/'
data_file = '/data/anau/SOD_classification/data/3_classes/propagated_0_0/stages.csv.20221114.3_classes.propagated.merged'

data_df = df = pd.read_csv(data_file, header=None)

# perform train-val-test split 70%-15%-15% split
X_train, X_test = train_test_split(data_df, test_size=0.3, random_state=1, shuffle=True)
X_val, X_test = train_test_split(X_test, test_size=0.5, random_state=1, shuffle=True)

print(data_df.shape, X_train.shape, X_val.shape, X_test.shape)

X_train.to_csv(data_path+'train_70', index=False, header=False)
X_val.to_csv(data_path+'val_15', index=False, header=False)
X_test.to_csv(data_path+'test_15', index=False, header=False)
