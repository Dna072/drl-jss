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

from custom_environment.dispatch_rules.environment_factory_dispatch_rules import (
    init_custom_factory_env,
)
from stable_baselines3.common.type_aliases import MaybeCallback
from custom_environment.dispatch_rules.environment_wrapper_dispatch_rules import (
    EnvWrapperDispatchRules,
)

# from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import DQN


class Agent:
    """
    DQN agent for learning the custom FactoryEnv environment
    """

    FILE_PATH_NAME: str = "./files/dqn_custom_factory_env"
    POLICY: str = (
        "MultiInputPolicy"  # converts multiple Dictionary inputs into a single vector
    )
    IS_VERBOSE: int = 1

    def __init__(self, custom_env: EnvWrapperDispatchRules) -> None:
        self.custom_env: EnvWrapperDispatchRules = custom_env
        print(f"Custom env jobs: {len(self.custom_env.get_pending_jobs())}")
        self.model: DQN = DQN(
            policy=self.POLICY,
            env=self.custom_env,
            verbose=self.IS_VERBOSE,
            learning_starts=10000,
        )

    def learn(
        self,
        total_time_steps: int = 200_000,
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
        import matplotlib.pyplot as plt

        obs, info = self.custom_env.reset()
        terminated, truncated = (False, False)
        rewards = []
        avg_machine_utilization = []
        avg_machine_idle_time = []
        steps = 0

        while steps < 10000:
            steps += 1
            print(f"steps: {steps}")
            action, _states = self.model.predict(observation=obs, deterministic=True)
            obs, reward, terminated, truncated, info = self.custom_env.step(
                action=action
            )

            rewards.append(reward)
            avg_machine_utilization.append(
                self.custom_env.get_average_machine_utilization_time_percentage()
            )
            avg_machine_idle_time.append(
                self.custom_env.get_average_machine_idle_time_percentage()
            )
            if terminated or truncated:
                obs, info = self.custom_env.reset()

        # plot rewards
        fig, axs = plt.subplots(3, 1)
        fig.suptitle("Shortest deadline first rule")
        axs[0].plot(rewards)
        axs[0].set_title("Rewards plot")
        axs[0].set(ylabel="Reward")

        axs[1].plot(avg_machine_utilization)
        axs[1].set_title("Avg Machine utilization")
        axs[1].set(ylabel="Avg Utilization %")

        axs[2].plot(avg_machine_idle_time)
        axs[2].set_title("Avg Machine idle time")
        axs[2].set(ylabel="Avg idleness %")

        for ax in axs.flat:
            ax.set(xlabel="Time step")

        plt.show()
        plt.savefig("../files/plots/evaluation_plot_1.png", format="png")


if __name__ == "__main__":
    from callback.plot_training_callback import PlotTrainingCallback

    plot_training_callback: PlotTrainingCallback = PlotTrainingCallback(plot_freq=100)

    custom_env = init_custom_factory_env(is_verbose=True)

    for job in custom_env.get_pending_jobs():
        print(job)
        print("----")

    agent = Agent(custom_env=custom_env)

    agent.learn(total_time_steps=300_000, log_interval=5, callback=None)
    # agent.learn()

    agent.save()

    # agent.load()
    agent.evaluate()
