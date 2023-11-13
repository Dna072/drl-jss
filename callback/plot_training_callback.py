from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt
import numpy as np


class PlotTrainingCallback(BaseCallback):
    """
    Callback subclass for plotting the training of an agent. Extends Stable_Baselines3 BaseCallback class.
    """

    CALLBACK_FREQ_REMAINDER = 0  # the remainder value representing the frequency of callback for plotting train data

    def __init__(self, plot_freq: int, verbose: int = 1) -> None:
        """
        Class constructor
        :param plot_freq: frequency of training data being plotted
        :param verbose: verbosity: 0 for no output, 1 for info messages, 2 for debug messages
        """
        super(PlotTrainingCallback, self).__init__(verbose)
        self.plot_freq: int = plot_freq
        self.rewards: list[int] = []
        self.num_eps: int = 0
        self.mean_returns: list[float] = []
        self.best_mean_return: float = -np.inf

    def _on_step(self) -> bool:
        """
        Called by model after each call to ``env.step()``.
        """
        self.rewards.append(self.training_env.get_attr("episode_reward_sum")[0])

        if self.n_calls % self.plot_freq == self.CALLBACK_FREQ_REMAINDER:
            self.mean_returns.append(
                np.mean(self.rewards) / self.num_eps if self.num_eps > 0 else 0
            )

            if self.mean_returns[-1] > self.best_mean_return:
                self.best_mean_return = self.mean_returns[-1]
                # TODO: save model every mean return improvement

            if self.verbose:
                print(f"Step: {self.n_calls}, Mean return: {self.mean_returns[-1]}")
            self.reset()
        return True

    def _on_rollout_end(self) -> None:
        """
        Increment episode counter on rollout end
        """
        self.num_eps += 1

    def _on_training_end(self) -> None:
        """
        Plot training data at end of agent training
        """
        self.plot_train_data()

    def reset(self) -> None:
        """
        Reset rewards array and episode counter
        """
        self.rewards = []
        self.num_eps = 0

    def plot_train_data(self) -> None:
        """
        Plot training data
        """
        plt.figure(figsize=(10, 6))
        plt.title("DQN Agent Training Performance")
        plt.xlabel("Steps")
        plt.ylabel("Mean Returns")
        plt.plot(self.mean_returns)
        plt.show()


if __name__ == "__main__":
    from custom_environment.environment_factory import init_custom_factory_env
    from custom_environment.environment import FactoryEnv
    from agent import Agent

    env: FactoryEnv = init_custom_factory_env()
    agent: Agent = Agent(custom_env=env)
    plot_training_callback: PlotTrainingCallback = PlotTrainingCallback(plot_freq=10)
    agent.learn(total_time_steps=100, log_interval=5, callback=plot_training_callback)
