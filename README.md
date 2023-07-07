# stage_of_human_decay_classification

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
Python3 and Tensorflow 2.0 are required. See requirements.txt for additional required packages/libraries/modules. 

### Installation
Clone the repo
   ```
   git clone https://github.com/annamarianau/stage_of_human_decay_classification.git
   ```
   
## To Run
Data preparation and label propagation
```
python3 01_label_preprocessing.py --input_file [path to input file] --output_file [path to output file]
python3 02_base_LPA.py OR 02_exact_label_propagation.sh  # if 02_base_LPA.py, run gen_embeddings.py first (optional)
python3 03_train_val_test_split.py --input_file [path to input file] --output_dir [path to output directory]
```

Train model - Two-step transfer learning
```
python3 train.py --config_path config/[folder_name]/[model_name].yaml --uid [unique for model, e.g., date] --process_data [0 for no and 1 for yes] 
```

Evaluate model
```
python3 test.py --config_path config/[folder_name]/[model_name]_tune.yaml --uid [unique for model, e.g., date] --process_data [0 for no and 1 for yes] 
```
