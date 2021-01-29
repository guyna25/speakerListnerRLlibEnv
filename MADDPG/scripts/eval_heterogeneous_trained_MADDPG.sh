#!/bin/sh

exp_name=heterogeneous_predators_2_predator_0_1_predator_1
model_name=maddpg_prey_maddpg_preds_3_on_1_1
model_episode=100000
scenario=allies/simple_tag_2_fixed
total_episodes=1000
save_rate=100
episode_len=1000
num_adversaries=3
num_fixed_adv=0
pred_0_network=0
pred_1_network=0
pred_2_network=1
pred_policy=maddpg
prey_policy=maddpg
save_dir=/home/asher/Homogeneous_Experiments/$model_name/$exp_name
load_dir=/home/asher/Desktop/ALLIES_public/coordinationlearning/data/spatial_distribution_models/$model_name/
model_file=$model_name'_'$model_episode

python ../experiments/train_hetrogeneous.py \
--exp-name $exp_name \
--scenario $scenario \
--num-fixed-adv $num_fixed_adv \
--num-adversaries $num_adversaries \
--pred-0-network $pred_0_network \
--pred-1-network $pred_1_network \
--pred-2-network $pred_2_network \
--good-policy $prey_policy \
--adv-policy $pred_policy \
--num-episodes $total_episodes \
--max-episode-len $episode_len \
--save-dir $save_dir \
--save-rate $save_rate \
--load-dir $load_dir \
--model-file $model_file \
--restore \
--testing \
--logging

echo Finished...
