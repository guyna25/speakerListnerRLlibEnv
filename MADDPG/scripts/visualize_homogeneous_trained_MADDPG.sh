#!/bin/sh

exp_name=homogeneous_predator_0
model_name=maddpg_prey_maddpg_preds_3_on_1_1
model_episode=100000
scenario=allies/simple_tag_2_fixed
total_episodes=1000
save_rate=100
episode_len=1000
num_adversaries=3
num_fixed_adv=0
pred_network=0
pred_policy=maddpg
prey_policy=maddpg
save_dir=/home/rolando/Homogeneous_Experiments/$model_name/$exp_name
load_dir=/home/rolando/GitLab/CONVERGE/Spatial_Distribution_Model_Data/$model_name/
model_file=$model_name'_'$model_episode

python ../experiments/train_homogeneous.py \
--scenario $scenario \
--max-episode-len $episode_len \
--num-episodes $total_episodes \
--num-adversaries $num_adversaries \
--num-fixed-adv $num_fixed_adv \
--pred-network $pred_network \
--good-policy $prey_policy \
--adv-policy $pred_policy \
--exp-name $exp_name \
--save-dir $save_dir \
--save-rate $save_rate \
--load-dir $load_dir \
--model-file $model_file \
--restore \
--testing \
--display

echo Finished...
