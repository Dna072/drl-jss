from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt
from os import makedirs, path
from torch import save
import numpy as np


def get_trendline(data: list[float] = None):
    episode_numbers = np.arange(1, len(data) + 1)
    coefficients = np.polyfit(episode_numbers, data, 5)
    return np.polyval(coefficients, episode_numbers)


class PlotTrainingCallback(BaseCallback):
    """
    Callback subclass for plotting the training of an agent. Extends Stable_Baselines3 BaseCallback class.
    """

    __CALLBACK_FREQ_REMAINDER = 0  # the remainder value representing the frequency of callback for plotting train data
    __FILE_PATH = "../files/models"
    __FILE_NAME = "best_mean_return.pt"

    def __init__(
        self, plot_freq: int, verbose: int = 1, is_save_best_model: bool = True, algorithm: str = 'DQN'
    ) -> None:
        """
        Class constructor
        :param plot_freq: frequency of training data being plotted
        :param verbose: verbosity: 0 for no output, 1 for info messages, 2 for debug messages
        :param is_save_best_model: conditional for if the model is saved every best means return
        """
        super(PlotTrainingCallback, self).__init__(verbose)
        # Config
        self.__plot_freq: int = plot_freq
        # Rewards
        self.__current_episode_rewards: list[float] = []  # Current episode rewards (gets reset when episode ends)
        self.__current_policy_rewards: list[float] = []  # Current policy rewards (gets reset on_rollout_end)
        self.__total_episodic_reward: list[float] = []  # Total reward for current episode
        self.__mean_episodic_reward: list[float] = []  # Mean reward per episode
        self.__total_policy_reward: list[float] = []  # Total reward for current policy
        self.__mean_policy_reward: list[float] = []  # Mean reward per policy
        # Tardiness
        self.__current_episode_tardiness: list[float] = []  # Current episode tardiness (gets reset when episode ends)
        self.__current_policy_tardiness: list[float] = []  # Current policy tardiness (gets reset on_rollout_end)
        self.__total_episodic_tardiness: list[float] = []  # Total tardiness for current episode
        self.__mean_episodic_tardiness: list[float] = []  # Mean tardiness per episode
        self.__total_policy_tardiness: list[float] = []  # Total tardiness for current policy
        self.__mean_policy_tardiness: list[float] = []  # Mean tardiness per policy

        # Initialisation and others
        self.__num_eps: int = 1
        self.__is_save_best_model: bool = is_save_best_model
        self.__mean_returns: list[float] = []
        self.__best_mean_return: float = -np.inf
        self.__algo: str = algorithm
        self.__num_policies: int = 0

        if not path.exists(path=self.__FILE_PATH):
            makedirs(name=self.__FILE_PATH)

    def _on_step(self) -> bool:
        """
        Called by model after each call to ``env.step()``.
        """
        self.__current_episode_rewards.append(self.training_env.get_attr("callback_step_reward")[0])
        self.__current_episode_tardiness.append(self.training_env.get_attr("callback_step_tardiness")[0])
        self.__current_policy_rewards.append(self.training_env.get_attr("callback_step_reward")[0])
        self.__current_policy_tardiness.append(self.training_env.get_attr("callback_step_tardiness")[0])

        if self.num_timesteps not in [0, 1] and self.training_env.get_attr("factory_time")[0] == 0:
            # I remove the last value that belongs to this new episode and add it to the next episode's array.
            last_c_reward = self.__current_episode_rewards.pop(-1)
            last_c_tardiness = self.__current_episode_tardiness.pop(-1)

            # Episodic Reward
            self.__total_episodic_reward.append(sum(self.__current_episode_rewards))
            self.__mean_episodic_reward.append(np.mean(self.__current_episode_rewards))
            # Episodic Tardiness
            self.__total_episodic_tardiness.append(sum(self.__current_episode_tardiness))
            self.__mean_episodic_tardiness.append(np.mean(self.__current_episode_tardiness))
            # Reset Episodic variables
            self.__current_episode_rewards = []
            self.__current_episode_tardiness = []
            # Add the values I've removed
            self.__current_episode_rewards.append(last_c_reward)
            self.__current_episode_tardiness.append(last_c_tardiness)
            # Increase episode number
            self.__num_eps += 1

        return True

    def _on_rollout_end(self) -> None:
        """
        In Stable-Baselines3, the _on_rollout_end method is part of the callback system
        and is executed at the end of each rollout during training. A rollout refers to a
        single trajectory where the agent interacts with the environment from the current
        state until a terminal state is reached. -Source: ChatGPT (To be confirmed)
        """
        # Reward
        self.__total_policy_reward.append(np.sum(self.__current_policy_rewards))
        self.__mean_policy_reward.append(np.mean(self.__current_policy_rewards))
        # Tardiness
        self.__total_policy_tardiness.append(np.sum(self.__current_policy_tardiness))
        self.__mean_policy_tardiness.append(np.mean(self.__current_policy_tardiness))

        # Every time we change policy, we check if our current model is better than the best, then we save it.
        # We might want to set a condition in order not to compare every time the policy changes
        # if self.__num_policies % self.__plot_freq == __CALLBACK_FREQ_REMAINDER:
        if self.__mean_policy_reward[-1] > self.__best_mean_return:
            self.__best_mean_return = self.__mean_policy_reward[-1]

            if self.__is_save_best_model:
                save(
                    obj=self.model.policy.state_dict(),
                    f=path.join(self.__FILE_PATH, self.__FILE_NAME),
                )
        self.reset()

    def _on_training_end(self) -> None:
        """
        Plot training data at end of agent training
        """
        self.plot_train_data()

    def reset(self) -> None:
        """
        Reset current policy rewards array and increase policy counter
        """
        self.__current_policy_tardiness = []
        self.__current_policy_rewards = []
        self.__num_policies += 1

    def plot_train_data(self) -> None:
        """
        Plot training data
        """

        # Plot Mean Episodic Rewards
        episode_numbers = np.arange(1, len(self.__mean_episodic_reward) + 1)
        plt.figure(figsize=(10, 6))
        plt.title(label=f"{self.__algo} Agent Mean Episodic Rewards")
        plt.xlabel(xlabel="Episode")
        plt.ylabel(ylabel="Mean Reward")
        # plt.plot(self.__mean_episodic_reward)
        plt.plot(episode_numbers, self.__mean_episodic_reward, label="Mean Reward")
        plt.plot(episode_numbers, get_trendline(self.__mean_episodic_reward), label="Trend Line", linestyle="--",
                 color="red")
        plt.legend()
        plt.savefig(f"./files/plots/{self.__algo}_training_mean_episodic_rewards.png", format="png")

        # Plot Mean Policy Rewards
        policy_numbers = np.arange(1, len(self.__mean_policy_reward) + 1)
        plt.figure(figsize=(10, 6))
        plt.title(label=f"{self.__algo} Agent Mean Policy Rewards")
        plt.xlabel(xlabel="Policy Nr")
        plt.ylabel(ylabel="Mean Reward")
        # plt.plot(self.__mean_policy_reward)
        plt.plot(policy_numbers, self.__mean_policy_reward, label="Mean Policy Reward")
        plt.plot(policy_numbers, get_trendline(self.__mean_policy_reward), label="Trend Line", linestyle="--",
                 color="red")
        plt.legend()
        plt.savefig(f"./files/plots/{self.__algo}_training_mean_policy_rewards.png", format="png")

        # Plot Mean Episodic Tardiness
        plt.figure(figsize=(10, 6))
        plt.title(label=f"{self.__algo} Agent Mean Episodic Tardiness")
        plt.xlabel(xlabel="Episode")
        plt.ylabel(ylabel="Mean Tardiness")
        # plt.plot(self.__mean_episodic_tardiness)
        plt.plot(episode_numbers, self.__mean_episodic_tardiness, label="Mean Reward")
        plt.plot(episode_numbers, get_trendline(self.__mean_episodic_tardiness), label="Trend Line", linestyle="--",
                 color="red")
        plt.legend()
        plt.savefig(f"./files/plots/{self.__algo}_training_mean_episodic_tardiness.png", format="png")

        # Plot Mean Policy Tardiness
        plt.figure(figsize=(10, 6))
        plt.title(label=f"{self.__algo} Agent Mean Policy Tardiness")
        plt.xlabel(xlabel="Policy Nr")
        plt.ylabel(ylabel="Mean Tardiness")
        # plt.plot(self.__mean_policy_tardiness)
        plt.plot(policy_numbers, self.__mean_policy_tardiness, label="Mean Policy Tardiness")
        plt.plot(policy_numbers, get_trendline(self.__mean_policy_tardiness), label="Trend Line", linestyle="--",
                 color="red")
        plt.legend()
        plt.savefig(f"./files/plots/{self.__algo}_training_mean_policy_tardiness.png", format="png")

        # There are also variables with total episodic rewards and tardiness and total policy rewards and tardiness
        # in case we want to plot them


if __name__ == "__main__":
    from custom_environment.environment_factory import init_custom_factory_env
    from custom_environment.environment import FactoryEnv
    from dqn_agent import Agent

    env: FactoryEnv = init_custom_factory_env()
    agent: Agent = Agent(custom_env=env)
    plot_training_callback: PlotTrainingCallback = PlotTrainingCallback(plot_freq=100)
    agent.learn(
        total_time_steps=100_000, log_interval=5, callback=plot_training_callback
    )
