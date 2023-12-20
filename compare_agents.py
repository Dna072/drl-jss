from fifo_agent import episodic_fifo_agent
from edd_agent import episodic_edd_agent
from random_agent import episodic_random_agent
from dqn_agent import episodic_dqn_agent, Agent as dqn_Agent
from ppo_agent import episodic_ppo_agent, Agent as ppo_Agent
import matplotlib.pyplot as plt
import numpy as np
from custom_environment.environment_factory import init_custom_factory_env
from custom_environment.utils import create_bins
from custom_environment.utils import save_agent_results, load_agent_results
import os

######
"""
In order to run this code, you will need to have the dqn agent trained and saved
Then add the path to the saved agent in the PATH constants below.
"""
######

N_EPISODES = 1_000
PLOT_GROUPING = N_EPISODES // 10
ENV_MAX_STEPS = 10_000
DQN_AGENT_PATH = "files/trainedAgents/dqn_agent_1100000"
PPO_AGENT_PATH = "files//trainedAgents/ppo_agent_1000000"
SAVE_PATH = "files/data/"

###################
#       EVAL      #
###################
'''
I added some control over the computations so that if it is already computed, it doesnt calculate everything it again
In case with a large number of episodes, this will save a lot of time.
'''
print("\033[96m"+"Starting Random"+"\033[0m")
RAND_PATH = SAVE_PATH + "random_data_" + str(N_EPISODES) + ".pkl"
if not os.path.exists(RAND_PATH):
    random_rewards, random_tardiness, random_jot, random_jnot = episodic_random_agent(n_episodes=N_EPISODES, env_max_steps=ENV_MAX_STEPS)
    save_agent_results(random_rewards, random_tardiness, random_jot, random_jnot, path= RAND_PATH)
else:
    random_rewards, random_tardiness, random_jot, random_jnot = load_agent_results(RAND_PATH)

print("\033[93m"+"Starting EDD"+"\033[0m")
EDD_PATH= SAVE_PATH + "edd_data_" + str(N_EPISODES) + ".pkl"
if not os.path.exists(EDD_PATH):
    edd_rewards, edd_tardiness, edd_jot, edd_jnot = episodic_edd_agent(n_episodes=N_EPISODES, env_max_steps=ENV_MAX_STEPS)
    save_agent_results(edd_rewards, edd_tardiness, edd_jot, edd_jnot, path= EDD_PATH)
else:
    edd_rewards, edd_tardiness, edd_jot, edd_jnot = load_agent_results(EDD_PATH)


print("\033[92m"+"Starting FIFO"+"\033[0m")
FIFO_PATH = SAVE_PATH + "fifo_data_" + str(N_EPISODES) + ".pkl"
if not os.path.exists(FIFO_PATH):
    fifo_rewards, fifo_tardiness, fifo_jot, fifo_jnot = episodic_fifo_agent(n_episodes=N_EPISODES, env_max_steps=ENV_MAX_STEPS)
    save_agent_results(fifo_rewards, fifo_tardiness, fifo_jot, fifo_jnot, path= FIFO_PATH)
else:
    fifo_rewards, fifo_tardiness, fifo_jot, fifo_jnot = load_agent_results(FIFO_PATH)

print("\033[96m"+"Starting DQN"+"\033[0m")
DQN_PATH = SAVE_PATH + "dqn_data_" + str(N_EPISODES) + ".pkl"
if not os.path.exists(DQN_PATH):
    agent = dqn_Agent(custom_env=init_custom_factory_env(max_steps=ENV_MAX_STEPS))
    agent.load(file_path_name=DQN_AGENT_PATH)
    dqn_rewards, dqn_tardiness, dqn_jot, dqn_jnot = agent.evaluate(num_of_episodes=N_EPISODES)
    save_agent_results(dqn_rewards, dqn_tardiness, dqn_jot, dqn_jnot, path= DQN_PATH)
else:
    dqn_rewards, dqn_tardiness, dqn_jot, dqn_jnot = load_agent_results(DQN_PATH)


print("\033[95m"+"Starting PPO"+"\033[0m")
PPO_PATH = SAVE_PATH + "ppo_data_" + str(N_EPISODES) + ".pkl"
if not os.path.exists(PPO_PATH):
    p_agent = ppo_Agent(custom_env=init_custom_factory_env(max_steps=ENV_MAX_STEPS))
    p_agent.load(file_path_name=PPO_AGENT_PATH)
    ppo_rewards, ppo_tardiness, ppo_jot, ppo_jnot = p_agent.evaluate(num_of_episodes=N_EPISODES)
    save_agent_results(ppo_rewards, ppo_tardiness, ppo_jot, ppo_jnot, path=PPO_PATH)
else:
    ppo_rewards, ppo_tardiness, ppo_jot, ppo_jnot = load_agent_results(PPO_PATH)

#############################
#         TARDINESS %       #
#############################
random_tardiness_performance = [a / (b + c) for a, b, c in zip(random_tardiness, random_jot, random_jnot)]
edd_tardiness_performance = [a / (b + c) for a, b, c in zip(edd_tardiness, edd_jot, edd_jnot)]
fifo_tardiness_performance = [a / (b + c) for a, b, c in zip(fifo_tardiness, fifo_jot, fifo_jnot)]
dqn_tardiness_performance = [a / (b + c) for a, b, c in zip(dqn_tardiness, dqn_jot, dqn_jnot)]
ppo_tardiness_performance = [a / (b + c) for a, b, c in zip(ppo_tardiness, ppo_jot, ppo_jnot)]

#############################
#         PLOT CONFIG       #
#############################
# Time steps
time_steps = np.arange(1, 11)  # 10 bins + 1
# Bar width for better visibility
bar_width = 0.15
# Set up positions for bars
random_positions = time_steps - 1.5 * bar_width
fifo_positions = time_steps - 0.5 * bar_width
edd_positions = time_steps + 0.5 * bar_width
dqn_positions = time_steps + 1.5 * bar_width
ppo_positions = time_steps + 2.5 * bar_width

#############################
#        RESHAPE DATA       #
#############################
random_rewards = create_bins(random_rewards, group_size=PLOT_GROUPING)
random_tardiness = create_bins(random_tardiness, group_size=PLOT_GROUPING)
edd_rewards = create_bins(edd_rewards, group_size=PLOT_GROUPING)
edd_tardiness = create_bins(edd_tardiness, group_size=PLOT_GROUPING)
fifo_rewards = create_bins(fifo_rewards, group_size=PLOT_GROUPING)
fifo_tardiness = create_bins(fifo_tardiness, group_size=PLOT_GROUPING)
dqn_rewards = create_bins(dqn_rewards, group_size=PLOT_GROUPING)
dqn_tardiness = create_bins(dqn_tardiness, group_size=PLOT_GROUPING)
ppo_rewards = create_bins(ppo_rewards, group_size=PLOT_GROUPING)
ppo_tardiness = create_bins(ppo_tardiness, group_size=PLOT_GROUPING)
# RESHAPE TARDINESS PERFORMANCE VECTORS
random_tardiness_performance = create_bins(random_tardiness_performance, group_size=PLOT_GROUPING)
edd_tardiness_performance = create_bins(edd_tardiness_performance, group_size=PLOT_GROUPING)
fifo_tardiness_performance = create_bins(fifo_tardiness_performance, group_size=PLOT_GROUPING)
dqn_tardiness_performance = create_bins(dqn_tardiness_performance, group_size=PLOT_GROUPING)
ppo_tardiness_performance = create_bins(ppo_tardiness_performance, group_size=PLOT_GROUPING)

#############################
#         PLOT REWARDS      #
#############################

# Plotting
plt.figure(figsize=(10, 6))
plt.bar(random_positions, random_rewards, width=bar_width, label='Random Rewards', color='#2ca02c')
plt.bar(fifo_positions, fifo_rewards, width=bar_width, label='FIFO Rewards', color='#d62728')
plt.bar(edd_positions, edd_rewards, width=bar_width, label='EDD Rewards', color='#9467bd')
plt.bar(dqn_positions, dqn_rewards, width=bar_width, label='DQN Rewards', color='#1f77b4')
plt.bar(ppo_positions, ppo_rewards, width=bar_width, label='PPO Rewards', color='#ff7f0e')

# Set y-axis to symlog scale
plt.yscale('symlog', linthresh=0.1)

# Customize the plot
plt.title('Comparative Plot for Average Episodic Reward (Symlog Scale)')
plt.xlabel("Episodes (x"+str(N_EPISODES//10)+")")
plt.ylabel('Rewards (Symlog Scale)')
plt.xticks(time_steps)
plt.legend()
plt.grid(True)
plt.show()

plt.savefig(f"./files/plots/Evaluation_Rewards_"+str(N_EPISODES)+".png", format="png")

#############################
#       PLOT TARDINESS      #
#############################
# Plotting
plt.figure(figsize=(10, 6))
plt.bar(random_positions, random_tardiness, width=bar_width, label='Random Tardiness', color='#2ca02c')
plt.bar(fifo_positions, fifo_tardiness, width=bar_width, label='FIFO Tardiness', color='#d62728')
plt.bar(edd_positions, edd_tardiness, width=bar_width, label='EDD Tardiness', color='#9467bd')
plt.bar(dqn_positions, dqn_tardiness, width=bar_width, label='DQN Tardiness', color='#1f77b4')
plt.bar(ppo_positions, ppo_tardiness, width=bar_width, label='PPO Tardiness', color='#ff7f0e')

# Customize the plot
plt.title('Comparative Plot for Average Episodic Tardiness')
plt.xlabel("Episodes (x"+str(N_EPISODES//10)+")")
plt.ylabel('Tardiness')
plt.xticks(time_steps)
plt.legend()
plt.grid(True)
plt.show()

plt.savefig(f"./files/plots/Evaluation_Tardiness_"+str(N_EPISODES)+".png", format="png")

#############################
#         TARDINESS %       #
#############################
# Plotting Tardiness per job completed
plt.figure(figsize=(10, 6))

plt.bar(random_positions, random_tardiness_performance, width=bar_width, label='Random Tardiness', color='#2ca02c')
plt.bar(fifo_positions, fifo_tardiness_performance, width=bar_width, label='FIFO Tardiness', color='#d62728')
plt.bar(edd_positions, edd_tardiness_performance, width=bar_width, label='EDD Tardiness', color='#9467bd')
plt.bar(dqn_positions, dqn_tardiness_performance, width=bar_width, label='DQN Tardiness', color='#1f77b4')
plt.bar(ppo_positions, ppo_tardiness_performance, width=bar_width, label='PPO Tardiness', color='#ff7f0e')

# Customize the plot
plt.title('Comparative Plot Average Episodic Tardiness/Jobs Completed')
plt.xlabel("Episodes (x"+str(N_EPISODES//10)+")")
plt.ylabel('Tardiness')
plt.xticks(time_steps)
plt.legend()
plt.grid(True)
plt.show()

plt.savefig(f"./files/plots/Evaluation_TardinessPercentage_"+str(N_EPISODES)+".png", format="png")

##############################
#     PLOT JOB COMPLETION    #
##############################

import matplotlib.pyplot as plt
import numpy as np

features = ("Jobs Completed On Time", "Jobs Completed Not On Time")
values = {
    'DQN': (round(np.mean(dqn_jot), 2), round(np.mean(dqn_jnot), 2)),
    'PPO': (round(np.mean(ppo_jot), 2), round(np.mean(ppo_jnot), 2)),
    'RAND': (round(np.mean(random_jot), 2), round(np.mean(random_jnot), 2)),
    'FIFO': (round(np.mean(fifo_jot), 2), round(np.mean(fifo_jnot), 2)),
    'EDD': (round(np.mean(edd_jot), 2), round(np.mean(edd_jnot), 2))
}

x = np.arange(len(features))  # the label locations
width = 0.15  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(layout='constrained')

for attribute, measurement in values.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute)
    ax.bar_label(rects, padding=3)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Nr of Jobs')
ax.set_title('Mean Job Completion per episode')
ax.set_xticks(x + width, features)
ax.legend(loc='upper right', ncols=3)
# ax.set_ylim(0, 250)
plt.show()

plt.savefig(f"./files/plots/Evaluation_JobsCompleted_"+str(N_EPISODES)+".png", format="png")