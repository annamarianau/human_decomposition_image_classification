# Script evaluates trained model on the test data. 
# To run: python3 test.py ./config/[config_file]
import os
import csv
import random
import pickle
import argparse
import yaml
import sys
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

tf.random.set_seed(123)

# load images and their labels, normalize images and categorize labels
def load_preprocess_data(config, file_w_paths):

    not_found = []
    img_paths = []
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
                
                img_paths.append(row[0].strip())
                data.append(img)
                labels.append(label)
            
            except:
                not_found.append(row)
            '''      
            counter += 1
            if counter == 50:
                break
            '''
    print('Images not found: ', not_found) 
    
    # save img files and ground truth as dataframe 
    df = pd.DataFrame(list(zip(img_paths, labels)), columns = ['img','gt'])

    # categorize labels
    labels_cat = tf.keras.utils.to_categorical(labels, num_classes=config['DATASET']['num_class'])
   
    # pickle the data
    f = open(file_w_paths+'.pickle', 'wb')
    pickle.dump((data, labels_cat, df), f)
    f.close()

    return data, labels_cat, df


def eval_metrics(gt, pred):
    CM = confusion_matrix(gt, pred)
    print('Confusion matrix:')
    print(CM)
    TP = CM.diagonal()
    FN = np.sum(CM, axis = 1) - TP
    FP = np.sum(CM, axis = 0) - TP
    
    P = []
    for i in range(len(TP)):
        if TP[i] != 0:
            prec = TP[i]/(TP[i] + FP[i])
            P.append(prec)
        else:
            P.append(0)
    print("Precision: ", P)
    avgP = np.mean(P)
    
    R = []
    for i in range(len(TP)):
        if TP[i] != 0:
            prec = TP[i]/(TP[i] + FN[i])
            R.append(prec)
        else:
            R.append(0)
    print("recall: ", R)
    avgR = np.mean(R)   
    maF1 = f1_score(gt, pred, average='macro')
    print("maF1: ", maF1)
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, required=True)
    parser.add_argument('--process_data', type=str, required=True, help='Does data need to be processed, "y" or "n"')
    args = parser.parse_args()
    config_path = args.config_path
    process_data = args.process_data
    
    # load config file
    with open(config_path) as file:
            config = yaml.safe_load(file)

    # load and process data, or just load if it has been previously processed and serialized
    if process_data == 'y':
        print('Loading and processing data from: ', config['DATASET']['test_path'])
        X_test, y_test, df_data = load_preprocess_data(config, config['DATASET']['test_path']) 
    elif process_data == 'n':
        print('Loading data from:  ', config['DATASET']['test_path'])
        f = open(config['DATASET']['test_path']+'.pickle', 'rb')
        test_data = pickle.load(f)
        X_test = test_data[0]
        y_test = test_data[1]
        df_data = test_data[2]
        f.close()
    
    # convert to numpy array
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    print('X_test.shape:',X_test.shape,'y_test.shape:', y_test.shape)

   # load trained model
    print('Loading model and predicting test set...')
    model = load_model(config['MODEL']['model_path'])
    print(model.summary())
    
    # predict test set
    prediction = model.predict(X_test) # each prediction is list of probabilities per class 
    df_data['pred'] = prediction.tolist()
    
    ## compute confusion matrix and eval metrics   
    print('Computing evaluation metrics...')
    
    # get ground truth 
    gt = list(df_data['gt'].values.astype(int))

    ### Top 1 ###
    print("### Top k=1 ###")
    pred_classes = list(prediction.argmax(axis=-1))
    df_data['k=1'] = pred_classes
    eval_metrics(gt, pred_classes)

    print()
    
    '''
    ### Top k>=1 ###
    pred_classes = []
    k_ls = [2]
    for k in k_ls:
        print("### Top k=", k, ' ###')
        for index, p in enumerate(prediction):  # p is for each image
            preds = p.argsort()[0-k:]  # the top k confident predictions
            added = False
            for p in preds:
                if p == gt[index]: # if one of the top k is the right prediction
                    pred_classes.append(p)
                    added = True
                    break
            if added == False:        
                pred_classes.append(preds[-1]) # the most confident
        
        df_data['k='+str(k)] = pred_classes
        eval_metrics(gt, pred_classes)
    '''
    
    # safe dataframe with columns: img_path,softmax_class_probs,ground_truth,prediction
    #df_data.to_csv(config['MODEL']['name']+'_predictions.csv', index=False)



