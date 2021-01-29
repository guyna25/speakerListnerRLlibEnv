#!/bin/sh

exp_name=spatial_distribution_models
model_name=maddpg_prey_maddpg_preds_3_on_1_5
model_episode=100000
scenario=allies/simple_tag_2_fixed
total_episodes=1
save_rate=100
episode_len=500
num_adversaries=3
num_fixed_adv=3
pred_policy=fixed
prey_policy=maddpg
save_dir=/home/asher/Desktop/ALLIES_public/coordinationlearning/data/$model_name/
load_dir=/home/asher/Desktop/ALLIES_public/coordinationlearning/data/$exp_name/$model_name/
plots_dir=/home/asher/Desktop/ALLIES_public/coordinationlearning/data/$model_name/
model_file=$model_name'_'$model_episode

python ../experiments/train_strategy_switch.py \
--exp-name $exp_name \
--scenario $scenario \
--num-fixed-adv $num_fixed_adv \
--num-adversaries $num_adversaries \
--switch-vector 50 100 150 200 250 300 400 \
--good-policy $prey_policy \
--adv-policy $pred_policy \
--num-episodes $total_episodes \
--max-episode-len $episode_len \
--save-dir $save_dir \
--plots-dir $plots_dir \
--load-dir $load_dir \
--model-file $model_file \
--restore \
--testing \
--logging

echo Finished...
