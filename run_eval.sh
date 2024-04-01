#!/bin/bash

# Define the path to the model
MODEL_PATH="./logs/240401_wine_mlp_protonet_3way_10shot_6query/best.model"

# Loop through seeds 0 to 99
for seed in {0..99}
do
    python eval.py --data_name "wine" --shot_num 10 --seed $seed --load_path "$MODEL_PATH"
done
