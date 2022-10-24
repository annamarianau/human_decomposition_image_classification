import os
import csv
import time
import random
import argparse
import yaml

import numpy as np
from keras.preprocessing import image


# build input pipeline 
def load_preprocess_data(file_w_img_paths):
    not_found = 0
    data = []
    labels = []
    # reading csv file 
    with open(file_w_img_paths, 'r') as csvfile:
        # create csv reader object 
        csv_reader = csv.reader(csvfile, delimiter = ",")
        # extracting each image one by one
        for row in csv_reader:
            label = row[1].strip()
            try:
                img = image.load_img(path=row[0].strip(),
                        target_size=(config['DATASET']['img_size'],
                        grayscale=False)
            
                img = image.img_to_array(img)  # convert PIL image instance to Numpy array 
                img = img/255  # normalize the values to range from 0 to 1
                data.append(img)
                labels.append(label)
            
            except:
                not_found += 1
                
    print(len(data), len(labels), not_found)
    print(data[0])
    print(labels[0])

    return data, labels

    
if __name__ == '__main__':

    # load config file
    config_path = sys.argv[1]
    with open(config_path) as file:
            config = yaml.safe_load(file)
    
    # load and preprocess data
    data, labels = load_preprocess_data(file_w_img_paths) 

    # compose the model: pretrained base model + new classification layers on top
    
    # train model



