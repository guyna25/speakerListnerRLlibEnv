#!/bin/sh

exp_name=maddpg_pred_prey
model_name=maddpg_pred_prey
scenario=allies/simple_tag_2_fixed
total_episodes=100000
save_rate=1000
episode_len=25
num_adversaries=3
pred_policy=maddpg
prey_policy=maddpg
save_dir=..\data\
load_dir=..\data\
plots_dir=..\data\plots\

python ../experiments/train.py \
--exp-name $exp_name \
--scenario $scenario \
--good-policy $prey_policy \
--adv-policy $pred_policy \
--num-adversaries $num_adversaries \
--num-episodes $total_episodes \
--max-episode-len $episode_len \
--save-dir $save_dir \
--plots-dir $plots_dir

echo Finished...