#!/bin/sh
attacker_level=5
defender_level=super

exp_name=hvt_level_$attacker_level"_attacker_vs_level_"$defender_level"_defender"
scenario=converge/simple_hvt_1v1_random
total_episodes=150000
save_rate=1000
episode_len=50
num_adversaries=1
pred_policy=maddpg
prey_policy=maddpg
load_dir=/Users/scottguan/CONVERGE/MADDGP-NB/MADDPG/data/level_k_data/



python ../experiments/load_level_k_agents.py \
--num-episodes $total_episodes \
--exp-name $exp_name \
--attacker-level $attacker_level \
--defender-level $defender_level \
--scenario $scenario \
--done-callback \
--good-policy $prey_policy \
--adv-policy $pred_policy \
--num-adversaries $num_adversaries \
--num-episodes $total_episodes \
--max-episode-len $episode_len \
--save-rate $save_rate \
--load-dir $load_dir \
--done-callback \
--restore \
--display

echo Finished...
