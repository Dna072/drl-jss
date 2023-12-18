from fifo_agent import episodic_fifo_agent
from edd_agent import episodic_edd_agent
from random_agent import episodic_random_agent
from dqn_agent import episodic_dqn_agent, Agent as dqn_Agent
from ppo_agent import episodic_ppo_agent, Agent as ppo_Agent
import matplotlib.pyplot as plt
import numpy as np
from custom_environment.environment_factory import init_custom_factory_env
from custom_environment.utils import create_bins
import pickle
######
"""
In order to run this code, you will need to have the dqn agent trained and saved
Then add the path to the saved agent in the PATH constants below.
"""
######

N_EPISODES = 100_000
PLOT_GROUPING = N_EPISODES//10
ENV_MAX_STEPS = 10_000
DQN_AGENT_PATH = "files/dqn_agent_5000000"
PPO_AGENT_PATH = "files/ppo_agent_1000000"
SAVE_PATH = "files/eval_data.pkl"

###################
#       EVAL      #
###################
print("Starting Random")
random_rewards, random_tardiness, random_jot, random_jnot = episodic_random_agent(n_episodes=N_EPISODES, env_max_steps=ENV_MAX_STEPS)
print("Starting edd")
edd_rewards, edd_tardiness, edd_jot, edd_jnot = episodic_edd_agent(n_episodes=N_EPISODES, env_max_steps=ENV_MAX_STEPS)
print("Starting fifo")
fifo_rewards, fifo_tardiness,fifo_jot, fifo_jnot = episodic_fifo_agent(n_episodes=N_EPISODES, env_max_steps=ENV_MAX_STEPS)
# dqn_rewards, dqn_tardiness, dqn_jot, dqn_jnot = episodic_dqn_agent(n_episodes=N_EPISODES,
#                                                 agent_path=DQN_AGENT_PATH,
#                                                 env_max_steps=ENV_MAX_STEPS
#                                                 )
print("Starting dqn")
agent = dqn_Agent(custom_env=init_custom_factory_env(max_steps=ENV_MAX_STEPS))
agent.load(file_path_name=DQN_AGENT_PATH)
dqn_rewards, dqn_tardiness, dqn_jot, dqn_jnot = agent.evaluate(num_of_episodes = N_EPISODES)
print("Starting Rppo")
p_agent = ppo_Agent(custom_env=init_custom_factory_env(max_steps=ENV_MAX_STEPS))
p_agent.load(file_path_name=PPO_AGENT_PATH)
ppo_rewards, ppo_tardiness, ppo_jot, ppo_jnot = p_agent.evaluate(num_of_episodes = N_EPISODES)

# ppo_rewards, ppo_tardiness,ppo_jot, ppo_jnot = episodic_ppo_agent(n_episodes=N_EPISODES,
#                                                 agent_path=PPO_AGENT_PATH,
#                                                 env_max_steps=ENV_MAX_STEPS
#                                                 )
#############################
#         TARDINESS %       #
#############################
random_tardiness_performance = [a / (b+c) for a, b, c in zip(random_tardiness, random_jot, random_jnot)]
edd_tardiness_performance = [a / (b+c) for a, b, c in zip(edd_tardiness, edd_jot, edd_jnot)]
fifo_tardiness_performance = [a / (b+c) for a, b, c in zip(fifo_tardiness, fifo_jot, fifo_jnot)]
dqn_tardiness_performance = [a / (b+c) for a, b, c in zip(dqn_tardiness, dqn_jot, dqn_jnot)]
ppo_tardiness_performance = [a / (b+c) for a, b, c in zip(ppo_tardiness, ppo_jot, ppo_jnot)]

#############################
#          SAVE DATA        #
#############################
data = {
    "random": (random_rewards, random_tardiness, random_jot, random_jnot),
    "edd": (edd_rewards, edd_tardiness, edd_jot, edd_jnot),
    "fifo": (fifo_rewards, fifo_tardiness, fifo_jot, fifo_jnot),
    "ppo": (ppo_rewards, ppo_tardiness, ppo_jot, ppo_jnot),
    "dqn": (dqn_rewards, dqn_tardiness, dqn_jot, dqn_jnot)
    }
with open(SAVE_PATH, "wb") as file:
    pickle.dump(data, file)

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
plt.bar(random_positions, random_rewards, width=bar_width, label='Random Rewards', color='green')
plt.bar(fifo_positions, fifo_rewards, width=bar_width, label='FIFO Rewards', color='red')
plt.bar(edd_positions, edd_rewards, width=bar_width, label='EDD Rewards', color='purple')
plt.bar(dqn_positions, dqn_rewards, width=bar_width, label='DQN Rewards', color='blue')
plt.bar(ppo_positions, ppo_rewards, width=bar_width, label='PPO Rewards', color='orange')


# Set y-axis to symlog scale
plt.yscale('symlog', linthresh=0.1)

# Customize the plot
plt.title('Comparative Plot for Reward at each Episode (Symlog Scale)')
plt.xlabel('Episode')
plt.ylabel('Rewards (Symlog Scale)')
plt.xticks(time_steps)
plt.legend()
plt.grid(True)
plt.show()

#############################
#       PLOT TARDINESS      #
#############################
# Plotting
plt.figure(figsize=(10, 6))
plt.bar(random_positions, random_tardiness, width=bar_width, label='Random Tardiness', color='green')
plt.bar(fifo_positions, fifo_tardiness, width=bar_width, label='FIFO Tardiness', color='red')
plt.bar(edd_positions, edd_tardiness, width=bar_width, label='EDD Tardiness', color='purple')
plt.bar(dqn_positions, dqn_tardiness, width=bar_width, label='DQN Tardiness', color='blue')
plt.bar(ppo_positions, ppo_tardiness, width=bar_width, label='PPO Tardiness', color='orange')

# Customize the plot
plt.title('Comparative Plot for Tardiness at the end of each Episode')
plt.xlabel('Episode')
plt.ylabel('Tardiness')
plt.xticks(time_steps)
plt.legend()
plt.grid(True)
plt.show()
# ------- MAYBE WE CAN APPLY THE SAME APPROACH BUT TO THE MEAN VALUES OVER 100'S OF EPISODES

#############################
#         TARDINESS %       #
#############################
# Plotting Tardiness per job completed
plt.figure(figsize=(10, 6))

plt.bar(random_positions, random_tardiness_performance, width=bar_width, label='Random Tardiness', color='green')
plt.bar(fifo_positions, fifo_tardiness_performance, width=bar_width, label='FIFO Tardiness', color='red')
plt.bar(edd_positions, edd_tardiness_performance, width=bar_width, label='EDD Tardiness', color='purple')
plt.bar(dqn_positions, dqn_tardiness_performance, width=bar_width, label='DQN Tardiness', color='blue')
plt.bar(ppo_positions, ppo_tardiness_performance, width=bar_width, label='PPO Tardiness', color='orange')

# Customize the plot
plt.title('Comparative Plot for Tardiness/Jobs Completed at the end of each Episode')
plt.xlabel('Episode')
plt.ylabel('Tardiness')
plt.xticks(time_steps)
plt.legend()
plt.grid(True)
plt.show()
# ------- MAYBE WE CAN APPLY THE SAME APPROACH BUT TO THE MEAN VALUES OVER 100'S OF EPISODES

##############################
#     PLOT JOB COMPLETION    #
##############################

import matplotlib.pyplot as plt
import numpy as np

features = ("Jobs completed on Time", "Jobs completed Not on time")
values = {
    'DQN': (np.mean(dqn_jot), np.mean(dqn_jnot)),
    'PPO': (np.mean(ppo_jot), np.mean(ppo_jnot)),
    'RAND': (np.mean(random_jot), np.mean(random_jnot)),
    'FIFO':(np.mean(fifo_jot), np.mean(fifo_jnot)),
    'EDD':(np.mean(edd_jot), np.mean(edd_jnot))
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