# # data from https://allisonhorst.github.io/palmerpenguins/
#
# import matplotlib.pyplot as plt
# import numpy as np
#
# algorithms = ("Jobs completed on Time", "Jobs completed Not on time")
# penguin_means = {
#     'DQN': (6, 0),
#     'PPO': (3, 2),
#     'RAND': (2, 5)
# }
#
# x = np.arange(len(algorithms))  # the label locations
# width = 0.25  # the width of the bars
# multiplier = 0
#
# fig, ax = plt.subplots(layout='constrained')
#
# for attribute, measurement in penguin_means.items():
#     offset = width * multiplier
#     rects = ax.bar(x + offset, measurement, width, label=attribute)
#     ax.bar_label(rects, padding=3)
#     multiplier += 1
#
# # Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_ylabel('Nr of Jobs')
# ax.set_title('Job Completion')
# ax.set_xticks(x + width, algorithms)
# ax.legend(loc='upper left', ncols=3)
# # ax.set_ylim(0, 250)
#
# plt.show()
import numpy as np
from dqn_agent import Agent
from custom_environment.environment_factory import init_custom_factory_env

def create_mean_array(input_array, group_size=10):
    # Reshape the array into a 2D array with the specified group size
    reshaped_array = np.reshape(input_array, (len(input_array) // group_size, group_size))

    # Calculate the mean along the second axis (axis=1)
    mean_array = np.mean(reshaped_array, axis=1)

    return mean_array

if __name__ == "__main__":
    from callback.plot_training_callback import PlotTrainingCallback
    LEARNING_MAX_STEPS = 80_100_000
    ENVIRONMENT_MAX_STEPS = 4_000
    JOBS_BUFFER_SIZE: int = 3
    N_MACHINES: int = 3
    N_RECIPES: int = 3
    GAMMA: float = 0.9
    plot_training_callback: PlotTrainingCallback = PlotTrainingCallback(plot_freq=10_000)

    agent = Agent(custom_env=init_custom_factory_env(max_steps=ENVIRONMENT_MAX_STEPS,
                                                     buffer_size=JOBS_BUFFER_SIZE,
                                              n_recipes=N_RECIPES, job_deadline_ratio=0.3, n_machines=N_MACHINES),
                  gamma=GAMMA,
                  exploration_fraction=0.58,
                  )

    # agent.load(file_path_name=f"files/trainedAgents/dqn_seco_{N_MACHINES}m_{N_RECIPES}r_{GAMMA}g_{JOBS_BUFFER_SIZE}b_j_que_ma_obs_x2_"+str(LEARNING_MAX_STEPS), exploration_initial_eps=0.08)
    agent.learn(
        total_time_steps=LEARNING_MAX_STEPS, log_interval=1000, callback=plot_training_callback
    )
    agent.save(file_path_name=f"files/trainedAgents/dqn_seco_{N_MACHINES}m_{N_RECIPES}r_{GAMMA}g_{JOBS_BUFFER_SIZE}b_j_que_ma_obs_"+str(LEARNING_MAX_STEPS))

    # agent.load(file_path_name=f"files/trainedAgents/dqn_seco_{N_MACHINES}m_{N_RECIPES}r_{GAMMA}g_{JOBS_BUFFER_SIZE}b_j_que_ma_obs_x3_"+str(LEARNING_MAX_STEPS))
    # agent.evaluate(num_of_episodes=1_000)
