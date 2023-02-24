# stage_of_human_decay_classification_leveraging

NIJ funded project focused on building a vision system to model stage of human decay leveraging a photographic collection documenting human decay.

### Directory structure/info
```
  .
  |-- config                           # config files to train and evaluate different CNN architectures
  |-- notebooks                        # notebooks to perform additional tasks (see notebooks for additional details)
  |-- 01_label_preprocessing           # DS to SOD label mapping      
  |-- 02_base_LPA.py                   # perform baseline label propagation
  |-- 02_exact_label_propagation.sh    # perform proposed label propagation
  |-- 03_train_val_test_split.py       # split data into train/val/test sets
  |-- gen_embeddings.py                # generate embeddigns prior to 02_base_LPA.py
  |-- train.py                         # Two-step transfer learning
  |-- test.py                          # Model evaluation on test data
  |-- run.sh                           # To run multiple experiments
  |-- requirements.txt
  |-- .gitignore
  |__ README.md
```

## Getting Started

### Prerequisites 
Some required packages/libraries:
* matplotlib==3.5.3
* numpy==1.23.2
* pandas==1.5.1
* PyYAML==6.0
* scikit_learn==1.1.3
* tensorflow==2.10.0

### Installation
Clone the repo
   ```
   git clone https://github.com/annamarianau/stage_of_human_decay_classification.git
   ```
   
## To Run

