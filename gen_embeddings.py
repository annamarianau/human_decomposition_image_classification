import cv2
import numpy as np
from tensorflow.keras.models import load_model
import csv
import sys
import os
from tensorflow.keras.preprocessing import image
import pandas as pd
import tensorflow as tf
import pickle
import random

model_name = '/da1_data/icputrd/decaying_human_body_part_classifier/models/inception/best_model_inception_epoch_-002-_acc_0.989001-_val_acc_0.98252_V2.h5'
model_type = 'inception'
model = load_model(model_name)
# remove last two layers of model and save as model2
x = model.layers[-3].output 
model2 = tf.keras.Model(inputs = model.input, outputs = x)

# import SOD-labeled data
df_w_labels = pd.read_csv('data/4_classes/stages.csv.20230201.annotated.multiple.4_classes', header=None, delimiter=',')
df_w_labels.columns = ['path', 'label']
# import unlabeled data
n = 74763 #number of records in file
s = 30000 #desired sample size
skip = sorted(random.sample(range(n),n-s))
df_no_labels = pd.read_csv('data/clusters.csv.multiple', delimiter=',', skiprows=skip, usecols=[0])
df_no_labels.columns = ['path']
df_no_labels['path'] = '/da1_data/icputrd/arf/mean.js/public' + df_no_labels['path'].astype(str)


# function to generate embeddings
def gen_embeddings(df, has_labels=False):
    inception_img_size = 299
    not_found = 0
    
    paths_ls = []
    embeddings_ls =[]
    labels_ls = []
    counter = 1
    for ind in df.index:
        img_data = []
        print(counter,  df['path'][ind])
        try:
            if model_type == 'resnet' or model_type == 'vgg':
                img = image.load_img( df['path'][ind].strip(),
                        target_size = (vgg_resnet_img_size, vgg_resnet_img_size, 3),
                        grayscale = False)
            elif model_type == 'inception':
                img = image.load_img( df['path'][ind].strip(),
                        target_size = (inception_img_size, inception_img_size, 3),
                        grayscale = False)
            img = image.img_to_array(img)
            img = img/255
            img_data.append(img)
            img_data_arr = np.array(img_data)

            prediction = model2.predict(img_data_arr)
            embeddings_ls.append(prediction[0])
            paths_ls.append(df['path'][ind])
            if has_labels:
                labels_ls.append(df['label'][ind])
            else:
                labels_ls.append(-1)
        except:
            not_found += 1
        
        counter += 1
    return paths_ls, embeddings_ls, labels_ls


# generate embeddings 
# get embeddings for df_w_labels
paths_ls1, embeddings_ls1, labels_ls1 = gen_embeddings(df_w_labels, has_labels=True)
print(len(paths_ls1), len(embeddings_ls1), len(labels_ls1))
with open('/data/anau/SOD_classification/data/4_classes/null_hypoth/paths_ls1', 'wb') as f:
    pickle.dump(paths_ls1, f)
with open('/data/anau/SOD_classification/data/4_classes/null_hypoth/embeddings_ls1', 'wb') as f:
    pickle.dump(embeddings_ls1, f)
with open('/data/anau/SOD_classification/data/4_classes/null_hypoth/labels_ls1', 'wb') as f:
    pickle.dump(labels_ls1, f)

# get embeddings for df_w_labels
paths_ls2, embeddings_ls2, labels_ls2 = gen_embeddings(df_no_labels, has_labels=False)
print(len(paths_ls2), len(embeddings_ls2), len(labels_ls2))
with open('/data/anau/SOD_classification/data/4_classes/null_hypoth/paths_ls2', 'wb') as f:
    pickle.dump(paths_ls2, f)
with open('/data/anau/SOD_classification/data/4_classes/null_hypoth/embeddings_ls2', 'wb') as f:
    pickle.dump(embeddings_ls2, f)
with open('/data/anau/SOD_classification/data/4_classes/null_hypoth/labels_ls2', 'wb') as f:
    pickle.dump(labels_ls2, f)

