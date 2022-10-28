# this script modifies the image path and labels
import csv
import re
import pandas as pd
from sklearn.model_selection import train_test_split

directory = '/home/anau/SOD_labeling/experiment_2/'
filename = 'stages.csv.20221027'
output_file = open('./data/' + filename + '.processed', "w")
classes = ['1','2','3','4','5','6','7','8','9','10','11','12','13']

with open(directory+filename, 'r') as csvfile:
    datareader = csv.reader(csvfile)
    for row in datareader:
        path = '/da1_data/icputrd/arf/mean.js/public'+row[0]
        path = re.sub('.icon', '', path)

        if (row[1] in classes) and (row[2] == 'head'):
            # modify image path
            path = '/da1_data/icputrd/arf/mean.js/public'+row[0]
            path = re.sub('.icon', '', path)
            
            # modify image label
            label = int(row[1])
            if 1 <= label <= 4:
                label = 0
            elif 5 <= label <= 7:
                label = 1
            elif 8 <= label <= 9:
                label = 2
            elif 10 <= label <= 13:
                label = 3    
            
            # write to new file
            #print(path + ',' + str(label))
            output_file.write(path + ',' + str(label) + '\n')

output_file.close()


# 70%-15%-15% split
data_df = df = pd.read_csv('./data/' + filename + '.processed', header=None)

X_train, X_test = train_test_split(data_df, test_size=0.3, random_state=1, shuffle=True)
X_val, X_test = train_test_split(X_test, test_size=0.5, random_state=1, shuffle=True)

print(data_df.shape, X_train.shape, X_val.shape, X_test.shape)

X_train.to_csv('./data/train_70', index=False, header=False)
X_val.to_csv('./data/val_15', index=False, header=False)
X_test.to_csv('./data/test_15', index=False, header=False)
