#!/bin/bash

###########################################
#               Create plots              #
###########################################
model_name=maddpg_hvt_1v1
DATA_PATH=/Users/scottguan/CONVERGE/MADDGP-NB/MADDPG/data/
MODEL=maddpg_hvt_1v1_custome_agent

# Call python script
python ../analysis/analyze_hvt_1v1_training_data.py \
--data-path "$DATA_PATH" \
--model $MODEL \
--num-episodes 300000
