import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle

max_level = 1

level_0_agrewards = pickle.load(open("../level_k_performance_data/maddpg_hvt_1v1_level_0_attacker_agrewards.pkl", 'rb'))
level_0_attacker_rewards = level_0_agrewards[0::2]
level_0_defender_rewards = level_0_agrewards[1::2]

plt.figure(figsize=(15,13))
plt.tight_layout(pad=3)

x_axis_label = '# of 1000 Episode Batches'
plt.subplot(121)
y_axis_label = 'Attacker Reward'
plt.plot(level_0_attacker_rewards, label="Level-0 (MADDPG)")
plt.ylabel(y_axis_label)
plt.xlabel(x_axis_label)
plt.title("Attacker Rewards")


plt.subplot(122)
y_axis_label = 'Defender Reward'
plt.plot(level_0_defender_rewards, label="Level-0 (MADDPG)")
plt.ylabel(y_axis_label)
plt.xlabel(x_axis_label)
plt.title("Defender Rewards")
plt.legend()

plt.subplot(223)
y_axis_label = 'Attacker Reward'
plt.ylabel(y_axis_label)
plt.xlabel(x_axis_label)
plt.title("Exploited Attacker Rewards")

plt.subplot(224)
y_axis_label = 'Defender Reward'
plt.ylabel(y_axis_label)
plt.xlabel(x_axis_label)
plt.title("Exploited Defender Rewards")
plt.legend()



for k in range(1, max_level+1):
    level_k_defender_agrewards = pickle.load(open("../level_k_performance_data/maddpg_hvt_1v1_level_{}_defender_agrewards.pkl".format(k),"rb"))
    level_k_attacker_agrewards = pickle.load(open("../level_k_performance_data/maddpg_hvt_1v1_level_{}_attacker_agrewards.pkl".format(k),"rb"))

    level_k_attacker_rewards = level_k_attacker_agrewards[0::2]
    exploited_level_K_1_defender_rewards = level_k_attacker_agrewards[1::2]
    level_k_defender_rewards = level_k_defender_agrewards[1::2]
    exploited_level_K_1_attacker_rewards = level_k_defender_agrewards[0::2]

    plt.subplot(221)
    plt.plot(level_k_attacker_rewards, label="Level-{} Attacker".format(k))
    plt.legend()

    plt.subplot(222)
    plt.plot(level_k_defender_rewards, label="Level-{} Defender".format(k))
    plt.legend()

    plt.subplot(223)
    plt.plot(exploited_level_K_1_attacker_rewards, label="Exploited Level-{} Attacker".format(k-1))
    plt.legend()

    plt.subplot(224)
    plt.plot(exploited_level_K_1_defender_rewards, label="Exploited Level-{} Defender".format(k-1))
    plt.legend()

plt.show()
