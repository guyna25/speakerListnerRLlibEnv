#!/bin/sh

exp_name=maddpg_hvt_1v1_21
model_name=maddpg_hvt_1v1_21
scenario=converge/simple_hvt_1v1_random
model_episode=200000
total_episodes=100
save_rate=1000
episode_len=50
pred_policy=maddpg
prey_policy=maddpg
save_dir=/home/rolando/GitLab/CONVERGE/MADDPG_HVT_Data/$model_name/
load_dir=/home/rolando/GitLab/CONVERGE/MADDPG_HVT_Data/$model_name/
plots_dir=/home/rolando/GitLab/CONVERGE/MADDPG_HVT_Data/$model_name/
model_file=$model_name'_'$model_episode

python ../experiments/train_hvt.py \
--scenario $scenario \
--done-callback \
--max-episode-len $episode_len \
--num-episodes $total_episodes \
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
