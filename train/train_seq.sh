#!/bin/bash


#python3 -u  train_cue.py --name='motion' --test_seq='5' --gt='True' 
#sleep 30


python3 -u train_cue.py --name='appearance' --test_seq='5' --gt='True'  --cnn_loss='triplet'
sleep 30

#python3 -u  train_cue.py --name='interaction' --test_seq='5' --gt='True' 
#sleep 30


python3 -u train_cue.py --name='target'    --test_seq='5' --gt='True'  --cnn_loss='triplet'

 