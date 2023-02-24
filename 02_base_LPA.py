import numpy as np
import csv
import sys
import pandas as pd
import pickle
from sklearn.datasets import load_digits
from sklearn.metrics import classification_report
from sklearn.semi_supervised import LabelPropagation
from time import perf_counter

def unpickle(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

paths_ls1 = unpickle('data/4_classes/null_hypoth/paths_ls1')
embeddings_ls1 = unpickle('data/4_classes/null_hypoth/embeddings_ls1')
labels_ls1 = unpickle('data/4_classes/null_hypoth/labels_ls1')
#print(len(paths_ls1), len(embeddings_ls1), len(labels_ls1))

paths_ls2 = unpickle('data/4_classes/null_hypoth/paths_ls2')
embeddings_ls2 = unpickle('data/4_classes/null_hypoth/embeddings_ls2')
labels_ls2 = unpickle('data/4_classes/null_hypoth/labels_ls2')
#print(len(paths_ls2), len(embeddings_ls2), len(labels_ls2))


embs_total = embeddings_ls1 + embeddings_ls2
#print(len(embs_total))

y_total = labels_ls1 + labels_ls2
#print(len(y_total)) 
unlabeled_set = np.arange(len(embeddings_ls1), len(embeddings_ls1)+len(embeddings_ls2), 1)
#print(len(unlabeled_set))

# label propagation
np.random.seed(0)

for k in [5,10,20]:
    start = perf_counter()
    print('k=', k)
    lp = LabelPropagation(kernel='knn', n_neighbors=k, max_iter=1000)
    lp.fit(embs_total, y_total) # run the algorithm we described above
    
    with open('/data/anau/SOD_classification/data/4_classes/null_hypoth/lpa_k_'+str(k), 'wb') as f:
        pickle.dump(lp.transduction_, f)

    # stop timer 
    print("Elapsed time:", perf_counter()-start)
    print(len(lp.transduction_))
