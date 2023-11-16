"""
DQN agent class for basic concept:
    1.  All jobs have only one recipe.
    2.  All jobs have a deadline.
    3.  All jobs are inserted into a machine using trays.
    4.  Machines can perform at least one recipe.
    5.  Machines have a maximum tray capacity as part of their attributes. Once a machine’s tray capacity is maxed out,
        no jobs can be assigned to it.
    6.  Recipes have a duration of time to complete.
    7.  The goal of the RL agent is to minimize tardiness (having most jobs completed before or on their deadline) and
        maximize efficiency of machines (machines should not be left idle for long periods).
    8.  The observation space is the job buffer (pending jobs to be scheduled), machines and their current capacity.
    9.  The action of RL agent to select which job, J_i to assign to machine M_m, where 0 <= i < |J| and 0 <= m < |M|.
        The action space is thus all possible combinations of (J, M) with addition of No-Op action (taken at a timestep)
    10. Only step when a machine is available, and maximum machines chosen at each step is one
"""

from custom_environment.environment_factory import init_custom_factory_env
from stable_baselines3.common.type_aliases import MaybeCallback
from custom_environment.environment import FactoryEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import DQN


class Agent:
    """
    DQN agent for learning the custom FactoryEnv environment
    """

    FILE_PATH_NAME: str = "files/dqn_custom_factory_env"
    POLICY: str = (
        "MultiInputPolicy"  # converts multiple Dictionary inputs into a single vector
    )
    IS_VERBOSE: int = 1

    def __init__(self, custom_env: FactoryEnv | Monitor) -> None:
        self.custom_env: FactoryEnv = custom_env
        self.model: DQN = DQN(
            policy=self.POLICY, env=self.custom_env, verbose=self.IS_VERBOSE
        )

    def learn(
        self,
        total_time_steps: int = 10_000,
        log_interval: int = 4,
        callback: MaybeCallback = None,
    ) -> None:
        self.model.learn(
            total_timesteps=total_time_steps,
            log_interval=log_interval,
            callback=callback,
        )

    def save(self, file_path_name: str = FILE_PATH_NAME) -> None:
        self.model.save(path=file_path_name)

    def load(self, file_path_name: str = FILE_PATH_NAME) -> None:
        self.model = DQN.load(path=file_path_name)

    def evaluate(self) -> None:
        obs, info = self.custom_env.reset()

        while True:
            action, _states = self.model.predict(observation=obs, deterministic=True)
            obs, reward, terminated, truncated, info = self.custom_env.step(
                action=action
            )

            if terminated or truncated:
                obs, info = self.custom_env.reset()


if __name__ == "__main__":
    from callback.plot_training_callback import PlotTrainingCallback

    # plot_training_callback: PlotTrainingCallback = PlotTrainingCallback(plot_freq=10)

    agent = Agent(custom_env=init_custom_factory_env())

    # agent.learn(total_time_steps=100, log_interval=5, callback=plot_training_callback)
    agent.learn()

    agent.save()

    agent.load()
    agent.evaluate()
