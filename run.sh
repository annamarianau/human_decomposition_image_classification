#!/bin/bash
# Script performs multiple experiments

python3 train.py --config_path config/4_classes/vgg16.yaml --process_data 'y'
python3 train.py --config_path config/4_classes/vgg16_tune.yaml --process_data 'n'
python3 test.py --config_path config/4_classes/vgg16.yaml --process_data 'y' > logs/4_classes/vgg16_lpa_k_5

echo ""

python3 train.py --config_path config/4_classes/resnet50.yaml --process_data 'y' 
python3 train.py --config_path config/4_classes/resnet50_tune.yaml --process_data 'n'
python3 test.py --config_path config/4_classes/resnet50.yaml --process_data 'y' > logs/4_classes/resnet50_lpa_k_5

echo ""

python3 train.py --config_path config/4_classes/inceptionV3.yaml --process_data 'y' 
python3 train.py --config_path config/4_classes/inceptionV3_tune.yaml --process_data 'n' 
python3 test.py --config_path config/4_classes/inceptionV3_tune.yaml --process_data 'y' > logs/4_classes/inceptionV3_lpa_k_5

echo ""

python3 train.py --config_path config/4_classes/xception.yaml --process_data 'y' 
python3 train.py --config_path config/4_classes/xception_tune.yaml --process_data 'n' 
python3 test.py --config_path config/4_classes/xception_tune.yaml --process_data 'y' logs/4_classes/xception_lpa_k_5
