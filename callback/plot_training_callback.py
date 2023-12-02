from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt
from os import makedirs, path
from torch import save
import numpy as np


class PlotTrainingCallback(BaseCallback):
    """
    Callback subclass for plotting the training of an agent. Extends Stable_Baselines3 BaseCallback class.
    """

    __CALLBACK_FREQ_REMAINDER = 0  # the remainder value representing the frequency of callback for plotting train data
    __FILE_PATH = "../files/models"
    __FILE_NAME = "best_mean_return.pt"

    def __init__(
        self, plot_freq: int, verbose: int = 1, is_save_best_model: bool = True
    ) -> None:
        """
        Class constructor
        :param plot_freq: frequency of training data being plotted
        :param verbose: verbosity: 0 for no output, 1 for info messages, 2 for debug messages
        :param is_save_best_model: conditional for if the model is saved every best means return
        """
        super(PlotTrainingCallback, self).__init__(verbose)
        self.__plot_freq: int = plot_freq
        self.__rewards: list[int] = []
        self.__num_eps: int = 0
        self.__mean_returns: list[float] = []
        self.__best_mean_return: float = -np.inf
        self.__is_save_best_model: bool = is_save_best_model

        if not path.exists(path=self.__FILE_PATH):
            makedirs(name=self.__FILE_PATH)

    def _on_step(self) -> bool:
        """
        Called by model after each call to ``env.step()``.
        """
        self.__rewards.append(self.training_env.get_attr("episode_reward_sum")[0])

        if self.n_calls % self.__plot_freq == self.__CALLBACK_FREQ_REMAINDER:
            self.__mean_returns.append(
                np.mean(self.__rewards) / self.__num_eps if self.__num_eps > 0 else 0
            )

            if self.__mean_returns[-1] > self.__best_mean_return:
                self.__best_mean_return = self.__mean_returns[-1]

                if self.__is_save_best_model:
                    save(
                        obj=self.model.policy.state_dict(),
                        f=path.join(self.__FILE_PATH, self.__FILE_NAME),
                    )

                if self.verbose:
                    print(
                        f"Step: {self.n_calls}, Mean return: {self.__mean_returns[-1]}"
                    )
            self.reset()
        return True

    def _on_rollout_end(self) -> None:
        """
        Increment episode counter on rollout end
        """
        self.__num_eps += 1

    def _on_training_end(self) -> None:
        """
        Plot training data at end of agent training
        """
        self.plot_train_data()

    def reset(self) -> None:
        """
        Reset rewards array and episode counter
        """
        self.__rewards = []
        self.__num_eps = 0

    def plot_train_data(self) -> None:
        """
        Plot training data
        """
        plt.figure(figsize=(10, 6))
        plt.title(label="DQN Agent Training Performance")
        plt.xlabel(xlabel="Steps")
        plt.ylabel(ylabel="Mean Returns")
        plt.plot(self.__mean_returns)
        #plt.show()
        plt.savefig('../files/plots/dqn_training.png', format='png')


if __name__ == "__main__":
    from custom_environment.environment_factory import init_custom_factory_env
    from custom_environment.environment import FactoryEnv
    from agent import Agent

    env: FactoryEnv = init_custom_factory_env()
    agent: Agent = Agent(custom_env=env)
    plot_training_callback: PlotTrainingCallback = PlotTrainingCallback(plot_freq=100)
    agent.learn(
        total_time_steps=100_000, log_interval=5, callback=plot_training_callback
    )
