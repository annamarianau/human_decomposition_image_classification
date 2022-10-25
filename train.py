import os
import csv
import time
import random
import argparse
import yaml
import sys

import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# build input pipeline 
def load_preprocess_data(config):
    file_w_paths = config['DATASET']['root_dataset']

    not_found = []
    data = []
    labels = []
    # reading csv file 
    with open(file_w_paths, 'r') as csvfile:
        # create csv reader object 
        csv_reader = csv.reader(csvfile, delimiter = ",")
        # extracting each image one by one
        counter = 0
        for row in csv_reader:
            label = row[1].strip()
            try:
                img = image.load_img(path=row[0].strip(),
                        target_size=(config['DATASET']['img_size'], config['DATASET']['img_size'], 3),
                        grayscale=False)
                img = image.img_to_array(img)  # convert PIL image instance to Numpy array 
                img = img/255  # normalize the values to range from 0 to 1
                data.append(img)
                labels.append(label)
            
            except:
                not_found.append(row)
            counter += 1
            if counter == 100:
                break
    
    print('Images not found: ', not_found)

    return data, labels

    
if __name__ == '__main__':

    # load config file
    config_path = sys.argv[1]
    with open(config_path) as file:
            config = yaml.safe_load(file)
    
    # load and preprocess data
    print('Loading and preprocessing data...')
    data, labels = load_preprocess_data(config) 
    
    # split into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(data, labels,
                                        random_state=123, shuffle=True,
                                        test_size=0.3)
    print(len(X_train), len(y_train), len(X_val), len(y_val))
    
    data_augmentation = tf.keras.Sequential([
    # preprocessing layer which randomly flips images during training.
    tf.keras.layers.RandomFlip('horizontal_and_vertical'),
    # preprocessing layer which randomly rotates images during training
    tf.keras.layers.RandomRotation(0.2)])
    
    # create the model = base model + new layers
    model_name = config['MODEL']['name'] 
    if model_name == 'resnet50':
        base_model = tf.keras.applications.ResNet50(include_top = False, weights='imagenet', 
                                            input_shape = (config['DATASET']['img_size'], 
                                                            config['DATASET']['img_size'],3),
                                            pooling = 'avg')

    base_model.trainable = False

    inp = tf.keras.Input((config['DATASET']['img_size'], config['DATASET']['img_size'], 3))
    x = data_augmentation(inp)
    x = base_model(x, training=False)
    #x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    out = tf.keras.layers.Dense(config['DATASET']['num_class'], activation='softmax')(x)

    model = tf.keras.Model(inp, out)
    
    # compile the model
    model.compile(optimizer='Adam',loss='categorical_crossentropy', metrics=['accuracy']) 
    print(model.summary())
    
    # create checkpoint
    checkpoint = tf.keras.callbacks.ModelCheckpoint(model_name + '_epoch_{epoch:03d}_acc_{acc:03f}_val_acc_{val_acc:.5f}.h5', 
                verbose=1, 
                monitor='val_acc', 
                save_best_only=True, 
                mode='auto')  

    

