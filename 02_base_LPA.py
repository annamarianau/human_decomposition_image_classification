# Script tp perform the baseline method, LPA.
# To run: python3 02_base_LPA.py (modify file paths below if needed)
# Note: It requires output from gen_embeddings.py which generates img embeddings
import numpy as np
import csv
import sys
import pandas as pd
import pickle
from sklearn.datasets import load_digits
from sklearn.metrics import classification_report
from sklearn.semi_supervised import LabelPropagation
from time import perf_counter

out_file = '/data/anau/SOD_classification/data/4_classes/null_hypoth/lpa_k_'  # modify if needed  

def unpickle(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

# modify lines if needed
# labeled img paths, embeddings, labels
paths_ls1 = unpickle('data/4_classes/null_hypoth/paths_ls1')
embeddings_ls1 = unpickle('data/4_classes/null_hypoth/embeddings_ls1')
labels_ls1 = unpickle('data/4_classes/null_hypoth/labels_ls1')

# unlabeled img paths, embeddings, labels (label is -1 to indicate unlabeled to LPA method)
paths_ls2 = unpickle('data/4_classes/null_hypoth/paths_ls2')
embeddings_ls2 = unpickle('data/4_classes/null_hypoth/embeddings_ls2')
labels_ls2 = unpickle('data/4_classes/null_hypoth/labels_ls2')

# total embeddings
embs_total = embeddings_ls1 + embeddings_ls2
y_total = labels_ls1 + labels_ls2
unlabeled_set = np.arange(len(embeddings_ls1), len(embeddings_ls1)+len(embeddings_ls2), 1)

# label propagation
np.random.seed(0)

# run LPA with varying number of neighbors k
for k in [5,10,20]:
    start = perf_counter()
    print('k=', k)
    lp = LabelPropagation(kernel='knn', n_neighbors=k, max_iter=1000)
    lp.fit(embs_total, y_total) # run the algorithm we described above
    
    with open(out_file+str(k), 'wb') as f:
        pickle.dump(lp.transduction_, f)

    # stop timer 
    print("Elapsed time:", perf_counter()-start)
    print(len(lp.transduction_))
