# To run: python3 train.py ./config/[config_file]
import os
import csv
import random
import argparse
import yaml
import sys
from datetime import datetime

import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

tf.random.set_seed(123)

# load images and their labels, normalize images and categorize labels
def load_preprocess_data(config, file_w_paths):

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
            ''' 
            counter += 1
            if counter == 100:
                break
            '''
    print('Images not found: ', not_found) 

    labels_cat = tf.keras.utils.to_categorical(labels, num_classes=config['DATASET']['num_class'])
    
    return data, labels_cat

    
if __name__ == '__main__':

    # load config file
    config_path = sys.argv[1]
    with open(config_path) as file:
            config = yaml.safe_load(file)
    
    # load and preprocess data
    print('Loading and preprocessing ', config['DATASET']['data_path'])
    X_train, y_train = load_preprocess_data(config, config['DATASET']['train_path']) 
    X_val, y_val = load_preprocess_data(config, config['DATASET']['val_path'])

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_val = np.array(X_val)
    y_val = np.array(y_val)

    print('X_train.shape:',X_train.shape,'y_train.shape:', y_train.shape)
    print('X_val.shape:', X_val.shape,'y_val.shape:', y_val.shape)

    #sys.exit(0)
    
    if config['TRAIN']['augment'] == True:
        data_augmentation = tf.keras.Sequential([
        # preprocessing layer which randomly flips images during training.
        #tf.keras.layers.RandomFlip('horizontal_and_vertical'),
        # preprocessing layer which randomly rotates images during training
        tf.keras.layers.RandomRotation(0.2)])
    
    # create base model or load pretrained model
    model_name = config['MODEL']['name'] 
    if model_name == 'resnet50':
        base_model = tf.keras.applications.ResNet50(include_top = False, weights='imagenet', 
                                            input_shape = (config['DATASET']['img_size'], 
                                                           config['DATASET']['img_size'],3))
    elif model_name == 'inceptionV3':
        base_model = tf.keras.applications.InceptionV3(include_top = False, weights='imagenet',
                                                            input_shape = (config['DATASET']['img_size'],
                                                            config['DATASET']['img_size'],3))
    
    elif model_name == 'inception_resnetV2':
        base_model = tf.keras.applications.InceptionResNetV2(include_top = False, weights='imagenet',
                                                            input_shape = (config['DATASET']['img_size'],
                                                            config['DATASET']['img_size'],3))
    else:
        model = tf.keras.models.load_model(config['MODEL']['tune_path'])
    

    # freez layers of base_model or make layers trainable if tuning 
    if config['TRAIN']['tune'] == False:
        base_model.trainable = False

        for i, layer in enumerate(base_model.layers):
            print(i, layer.name,"-", layer.trainable)
    else:
        for layer in model.layers:
            layer.trainable = True

        for i, layer in enumerate(model.layers):
            print(i, layer.name,"-", layer.trainable)

    
    # define model architecture
    if config['TRAIN']['tune'] == False:
        inp = tf.keras.Input((config['DATASET']['img_size'], config['DATASET']['img_size'], 3))
        if config['TRAIN']['augment'] == True:
            x = data_augmentation(inp)
            x = base_model(x, training=False)
        else:
            x = base_model(inp, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        #x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        out = tf.keras.layers.Dense(config['DATASET']['num_class'], activation='softmax')(x)

        model = tf.keras.Model(inp, out)
        
        # compile the model
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config['TRAIN']['lr']),
                    loss='categorical_crossentropy',
                    metrics=['accuracy']) 
    
    print(model.summary())
        
    # create checkpoint and early stopping 
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
                filepath='checkpoints/' + model_name + '_{epoch:03d}-{accuracy:.3f}-{val_accuracy:.3f}.h5',
                monitor='val_accuracy',
                verbose=1,
                save_best_only=True,
                save_weights_only=True,
                save_freq='epoch')  

    early_stop = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', 
                patience=50,
                mode='min', 
                verbose=1) 
    
    
    # train model
    print('Training ',  config['MODEL']['name'])
    batch_size = config['TRAIN']['batch_size']
    num_epoch = config['TRAIN']['num_epoch']
    train_steps = X_train.shape[0] // batch_size
    val_steps = X_val.shape[0] // batch_size
    
    #print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print('num_epochs:', num_epoch, 'batch_size:', batch_size, 'train_steps:', train_steps, 
            'val_steps:', val_steps)

    # fit model
    history = model.fit(X_train,
                        y_train,
                        steps_per_epoch=train_steps,
                        validation_data=(X_val, y_val),
                        validation_steps=val_steps,
                        batch_size=batch_size,
                        epochs=num_epoch,
                        callbacks=[checkpoint, early_stop],
                        verbose=1) 
   
    # save model
    print('Saving to ', config['MODEL']['model_path'])
    model.save(filepath=config['MODEL']['model_path'], save_format="h5")
    
    #sys.exit(0)
    
    # model performance 
    # store results
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    # plot results
    # accuracy
    plt.figure(figsize=(10, 16))
    plt.rcParams['figure.figsize'] = [16, 9]
    plt.rcParams['font.size'] = 14
    plt.rcParams['axes.grid'] = True
    plt.rcParams['figure.facecolor'] = 'white'
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.title(f'Training and Validation Accuracy')
    plt.savefig('plots/'+model_name+'_acc')
 
    # loss
    plt.figure(figsize=(10, 16))
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.title(f'Training and Validation Loss')
    plt.xlabel('epoch')
    plt.tight_layout(pad=3.0)
    plt.savefig('plots/'+model_name+'_loss')


