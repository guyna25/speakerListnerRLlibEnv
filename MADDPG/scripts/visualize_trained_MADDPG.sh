#!/bin/sh

exp_name=spatial_distribution_models
model_name=maddpg_prey_maddpg_preds_3_on_1_5
model_episode=100000
scenario=allies/simple_tag_2_fixed
total_episodes=10
save_rate=5
episode_len=500
num_adversaries=3
pred_policy=fixed
prey_policy=maddpg
save_dir=/home/asher/Desktop/ALLIES/coordinationlearning/data/$model_name/
load_dir=/home/asher/Desktop/ALLIES/coordinationlearning/data/$exp_name/$model_name/
plots_dir=/home/asher/Desktop/ALLIES/coordinationlearning/data/$model_name/
model_file=$model_name'_'$model_episode

python ../experiments/train.py \
--scenario $scenario \
--max-episode-len $episode_len \
--num-episodes $total_episodes \
--num-adversaries $num_adversaries \
--num-fixed-adv 3 \
--good-policy $prey_policy \
--adv-policy $pred_policy \
--exp-name $exp_name \
--save-dir $save_dir \
--save-rate $save_rate \
--load-dir $load_dir \
--model-file $model_file \
--plots-dir $plots_dir \
--restore \
--testing \
--display

echo Finished...
