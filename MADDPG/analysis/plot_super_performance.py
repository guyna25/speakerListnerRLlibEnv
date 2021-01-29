import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle



super_agrewards = pickle.load(open("../data_super_agent/super_defender/super_defender_max_level_5_agrewards.pkl", 'rb'))
super_defender_rewards = super_agrewards[1::2]

plt.figure(figsize=(8,7))
plt.tight_layout(pad=3)

x_axis_label = '# of 1000 Episode Batches'
plt.plot()
y_axis_label = 'Attacker Reward'
plt.plot(super_defender_rewards, label="Super Defender")
plt.ylabel(y_axis_label)
plt.xlabel(x_axis_label)
plt.title("Super Defender Rewards")
plt.ylim((-15,0))

plt.show()
