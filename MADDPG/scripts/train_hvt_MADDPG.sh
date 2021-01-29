#!/bin/sh

exp_name=maddpg_hvt_1v1
model_name=maddpg_hvt_1v1
scenario=converge/simple_hvt_1v1_random
total_episodes=150000
save_rate=1000
episode_len=50
num_adversaries=1
customized_index=0
pred_policy=maddpg
prey_policy=maddpg
save_dir=/Users/scottguan/CONVERGE/MADDGP-NB/MADDPG/data/$model_name/
load_dir=/Users/scottguan/CONVERGE/MADDGP-NB/MADDPG/data/$model_name/
plots_dir=/Users/scottguan/CONVERGE/MADDGP-NB/MADDPG/data/$model_name/plots/


python ../experiments/train_hvt.py \
--exp-name $exp_name \
--model-name $model_name \
--scenario $scenario \
--done-callback \
--log-loss \
--good-policy $prey_policy \
--adv-policy $pred_policy \
--num-adversaries $num_adversaries \
--num-episodes $total_episodes \
--max-episode-len $episode_len \
--save-rate $save_rate \
--save-dir $save_dir \
--plots-dir $plots_dir \


echo Finished...
