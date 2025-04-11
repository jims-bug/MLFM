#!/bin/bash
# * laptop
# * MLFM with Bert
python ./train.py --model_name MLFMbert --dataset laptop --vocab_dir ./dataset/Laptops_corenlp --seed 1000 --num_epoch 20 --adaptiveCrossFusion_num_iterations 1 --cuda 0  

# * restaurant
# * MLFM with Bert
python ./train.py --model_name MLFMbert --dataset restaurant --vocab_dir ./dataset/Restaurants_corenlp --seed 1000 --num_epoch 15 --adaptiveCrossFusion_num_iterations 2 --cuda 0 

# * twitter
# * MLFM with Bert
python ./train.py --model_name MLFMbert --dataset twitter --vocab_dir ./dataset/Tweets_corenlp --seed 1000 --num_epoch 15 --adaptiveCrossFusion_num_iterations 2 --cuda 0 

# * MAMS
# * MLFM with Bert
python ./train.py --model_name MLFMbert --dataset mams --vocab_dir ./dataset/MAMS_corenlp --seed 1000 --num_epoch 15 --adaptiveCrossFusion_num_iterations 2 --cuda 0 

# * restaurant15
# * MLFM with Bert
python ./train.py --model_name MLFMbert --dataset restaurant15 --vocab_dir ./dataset/semeval15 --seed 1000 --num_epoch 15 --adaptiveCrossFusion_num_iterations 2 --cuda 0 

# * restaurant16
# * MLFM with Bert
python ./train.py --model_name MLFMbert --dataset restaurant16 --vocab_dir ./dataset/semeval16 --seed 1000 --num_epoch 15 --adaptiveCrossFusion_num_iterations 2 --cuda 0 



