#!/bin/sh

exp_name=maddpg_hvt_1v1
model_name=maddpg_hvt_1v1
scenario=converge/simple_hvt_1v1_random
total_episodes=150000
save_rate=1000
episode_len=100
num_adversaries=1
pred_policy=maddpg
prey_policy=maddpg
save_dir=/Users/scottguan/CONVERGE/MADDGP-NB/MADDPG/data/$model_name/
load_dir=/Users/scottguan/CONVERGE/MADDGP-NB/MADDPG/data/$model_name/
plots_dir=/Users/scottguan/CONVERGE/MADDGP-NB/MADDPG/data/$model_name/plots/
# model_file=$model_name'_'$total_episodes
model_file=$model_name'_150000'

python ../experiments/train_hvt.py \
--exp-name $exp_name \
--scenario $scenario \
--load-dir $load_dir \
--model-file $model_file \
--max-episode-len $episode_len \
--done-callback \
--restore \
--testing \
--display


echo Finished...
