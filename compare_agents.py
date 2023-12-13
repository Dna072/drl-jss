from fifo_agent import episodic_fifo_agent
from edd_agent import episodic_edd_agent
from random_agent import episodic_random_agent
from dqn_agent import episodic_dqn_agent
from ppo_agent import episodic_ppo_agent
import matplotlib.pyplot as plt
import numpy as np

######
"""
In order to run this code, you will need to have the dqn agent trained and saved
"""
######

N_EPISODES = 10
ENV_MAX_STEPS = 10_000
DQN_AGENT_PATH = "files/dqn_custom_factory_env_2"
PPO_AGENT_PATH = "files/ppo_custom_factory_env"

###################
#       EVAL      #
###################
random_rewards, random_tardiness = episodic_random_agent(n_episodes=N_EPISODES, env_max_steps=ENV_MAX_STEPS)
edd_rewards, edd_tardiness = episodic_edd_agent(n_episodes=N_EPISODES, env_max_steps=ENV_MAX_STEPS)
fifo_rewards, fifo_tardiness = episodic_fifo_agent(n_episodes=N_EPISODES, env_max_steps=ENV_MAX_STEPS)
dqn_rewards, dqn_tardiness = episodic_dqn_agent(n_episodes=N_EPISODES,
                                                agent_path=DQN_AGENT_PATH,
                                                env_max_steps=ENV_MAX_STEPS
                                                )
ppo_rewards, ppo_tardiness = episodic_ppo_agent(n_episodes=N_EPISODES,
                                                agent_path=PPO_AGENT_PATH,
                                                env_max_steps=ENV_MAX_STEPS
                                                )

###################
#       PLOT      #
###################
# Time steps
time_steps = np.arange(1, N_EPISODES+1)
# Bar width for better visibility
bar_width = 0.2
# Set up positions for bars
random_positions = time_steps - 1.5 * bar_width
fifo_positions = time_steps - 0.5 * bar_width
edd_positions = time_steps + 0.5 * bar_width
dqn_positions = time_steps + 1.5 * bar_width
ppo_positions = time_steps + 2.5 * bar_width

# Plotting
plt.figure(figsize=(10, 6))
plt.bar(random_positions, random_rewards, width=bar_width, label='Random Rewards')
plt.bar(fifo_positions, fifo_rewards, width=bar_width, label='FIFO Rewards')
plt.bar(edd_positions, edd_rewards, width=bar_width, label='EDD Rewards')
plt.bar(dqn_positions, dqn_rewards, width=bar_width, label='DQN Rewards', color='orange')
plt.bar(ppo_positions, ppo_rewards, width=bar_width, label='PPO Rewards', color='purple')

# Set y-axis to symlog scale
plt.yscale('symlog', linthresh=0.1)

# Customize the plot
plt.title('Comparative Plot for Reward at each Time Step (Symlog Scale)')
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
plt.bar(random_positions, random_tardiness, width=bar_width, label='Random Tardiness')
plt.bar(fifo_positions, fifo_tardiness, width=bar_width, label='FIFO Tardiness')
plt.bar(edd_positions, edd_tardiness, width=bar_width, label='EDD Tardiness')
plt.bar(dqn_positions, dqn_tardiness, width=bar_width, label='DQN Tardiness', color='orange')
plt.bar(ppo_positions, ppo_tardiness, width=bar_width, label='PPO Tardiness', color='purple')

# Customize the plot
plt.title('Comparative Plot for Tardiness at each Time Step')
plt.xlabel('Episode')
plt.ylabel('Tardiness')
plt.xticks(time_steps)
plt.legend()
plt.grid(True)
plt.show()
# ------- MAYBE WE CAN APPLY THE SAME APPROACH BUT TO THE MEAN VALUES OVER 100'S OF EPISODES
