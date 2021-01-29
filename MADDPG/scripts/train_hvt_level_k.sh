#!/bin/sh
level=0
training_role=attacker
exp_name=maddpg_hvt_1v1_level_$level"_"$training_role
model_name=level_$level"_"$training_role
scenario=converge/simple_hvt_1v1_model17
total_episodes=150000
save_rate=1000
episode_len=50
num_adversaries=1
pred_policy=maddpg
prey_policy=maddpg
save_dir=/Users/scottguan/CONVERGE/MADDGP-NB/MADDPG/data_model17/$model_name/
load_dir=/Users/scottguan/CONVERGE/MADDGP-NB/MADDPG/data_model17/$model_name/
plots_dir=/Users/scottguan/CONVERGE/MADDGP-NB/MADDPG/data_model17/$model_name/
model_file="maddpg_hvt_1v1_level_4_attacker_1050000"

python ../experiments/train_hvt_level_k.py \
--num-episodes $total_episodes \
--plots-dir $plots_dir \
--training-role $training_role \
--exp-name $exp_name \
--level $level \
--scenario $scenario \
--done-callback \
--good-policy $prey_policy \
--adv-policy $pred_policy \
--num-adversaries $num_adversaries \
--num-episodes $total_episodes \
--max-episode-len $episode_len \
--save-rate $save_rate \
--save-dir $save_dir \
--load-dir $load_dir \
--plots-dir $plots_dir \
--model-file $model_file \
--done-callback \
--logging \
--log-loss \
# --restore \


echo Finished...
