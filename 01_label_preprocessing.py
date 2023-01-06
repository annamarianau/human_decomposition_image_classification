# This script preprocess the images paths and their labels. Labels tranformed using different stage of decay ranges.
import csv
import re
import pandas as pd
from sklearn.model_selection import train_test_split

directory = '/home/anau/SOD_labeling/experiment_2/processed/' 
filename = 'stages.csv.20221114_correct'  # SOD-labeled data was processed
output_file = open('./data/3_classes/' + filename + '.3_classes', "w")  # modify if needed
classes = ['1','2','3','4','5','6','7','8','9','10','11','12','13']

# preprocess images paths and labels 
with open(directory+filename, 'r') as csvfile:
    datareader = csv.reader(csvfile)
    for row in datareader:

        if (row[1] in classes) and (row[2] == 'head'):
            # modify image path
            path = '/da1_data/icputrd/arf/mean.js/public'+row[0]
            path = re.sub('.icon', '', path)
            
            # modify image label
            label = int(row[1])
            if 1 <= label <= 4:
                label = 0
            '''            
            elif 5 <= label <= 7:
                label = 1
            elif 8 <= label <= 9:
                label = 2
            elif 10 <= label <= 13:
                label = 3   
            '''
            if 5 <= label <= 9:
                label = 1
            elif 10 <= label <= 13:
                label = 2
            
            # write to new file
            #print(path + ',' + str(label))
            output_file.write(path + ',' + str(label) + '\n')

output_file.close()
