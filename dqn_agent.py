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
import numpy as np
import matplotlib.pyplot as plt
from custom_environment.utils import print_observation, print_jobs


class Agent:
    """
    DQN agent for learning the custom FactoryEnv environment
    """

    FILE_PATH_NAME: str = "files/dqn_custom_factory_env_2"
    POLICY: str = (
        "MultiInputPolicy"  # converts multiple Dictionary inputs into a single vector
    )
    IS_VERBOSE: int = 1

    def __init__(self, custom_env: FactoryEnv | Monitor) -> None:
        self.custom_env: FactoryEnv = custom_env
        self.model: DQN = DQN(
            policy=self.POLICY, env=self.custom_env, verbose=self.IS_VERBOSE, learning_starts=20000,
            learning_rate=1e-3, gamma=0.6, exploration_fraction=0.25
        )

    def learn(
        self,
        total_time_steps: int = 10_000,
        log_interval: int = 4,
        callback: MaybeCallback = None,
    ) -> None:
        self.custom_env.set_termination_reward(-1000)
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
        self.custom_env.set_termination_reward(-100000)
        num_of_episodes = 100
        avg_returns_per_episode = []
        tardiness_per_episode = []
        job_times_past_deadline = []
        jobs_completed_on_time = []
        jobs_completed_late = []
        avg_tardiness_of_late_jobs_per_episode = []
        returns = []
        episode = 0
        steps = 0

        while episode < num_of_episodes:
            print_jobs(env=self.custom_env)
            print_observation(obs, nr_machines=len(self.custom_env.get_machines()))

            action, _states = self.model.predict(observation=obs, deterministic=True)
            # action = np.random.randint(low=0, high=7)
            print(f'Take action: {action}')

            obs, reward, terminated, truncated, info = self.custom_env.step(
                action=action
            )
            print(f"reward: {reward}")
            print(f"info: {info}")

            print_observation(obs, nr_machines=len(self.custom_env.get_machines()))
            print(f"reward: {reward}")
            print(f"info: {info}")

            # test = input("Pause to read")

            returns.append(reward)
            steps += 1

            if terminated or truncated:
                print(f"avg return: { np.sum(returns) / steps}")
                avg_returns_per_episode.append(np.sum(returns) / steps)
                tardiness_per_episode.append(self.custom_env.get_tardiness_percentage())
                job_times_past_deadline.append(self.custom_env.get_avg_time_past_deadline())
                jobs_completed_late.append(info['JOBS_NOT_COMPLETED_ON_TIME'])
                jobs_completed_on_time.append(info['JOBS_COMPLETED_ON_TIME'])
                avg_tardiness_of_late_jobs_per_episode.append(info['AVG_TARDINESS_OF_LATE_JOBS'])
                steps = 0
                returns = []
                obs, info = self.custom_env.reset()
                episode += 1

        # plt.figure(figsize=(10, 6))
        # plt.plot(avg_returns_per_episode)
        # plt.ylabel('Avg returns')
        # plt.xlabel('Episodes')
        # plt.suptitle('Evaluation: Avg returns per episode')
        # plt.show()
        print(f"Avg jobs completed on time: {sum(jobs_completed_on_time) / len(jobs_completed_on_time)}")
        print(f"Avg jobs completed late: {sum(jobs_completed_late) / len(jobs_completed_late)}")
        print(f"Avg tardiness of late jobs over episodes: "
              f"{sum(avg_tardiness_of_late_jobs_per_episode) / len(avg_tardiness_of_late_jobs_per_episode)}")

        fig, axs = plt.subplots(3, 1)
        fig.suptitle("Avg returns per episode")
        axs[0].plot(avg_returns_per_episode)
        axs[0].set_title("Rewards plot")
        axs[0].set(ylabel="Avg returns")

        axs[1].plot(tardiness_per_episode)
        axs[1].set_title("Tardiness plot")
        axs[1].set(ylabel="Tardiness percentage")

        axs[2].plot(job_times_past_deadline)
        axs[2].set_title("Times past deadline")
        axs[2].set(ylabel="Time")

        for ax in axs.flat:
            ax.set(xlabel="Episode")

        for ax in axs.flat:
            ax.label_outer()

        plt.suptitle('DQN Evaluation')
        plt.show()


def episodic_dqn_agent(n_episodes: int = 10, agent_path: str = "files/dqn_custom_factory_env_2",
                       env_max_steps: int = 100):
    ep_reward = []
    ep_tardiness = []
    dqn_agent = Agent(custom_env=init_custom_factory_env(max_steps=env_max_steps, is_evaluation=True))
    dqn_agent.load(agent_path)
    for e in range(n_episodes):
        env = init_custom_factory_env(is_verbose=False)
        obs, info = env.reset()
        tot_reward = 0
        curr_tardiness = []
        while 1:  # the environment has its own termination clauses, so it will trigger the break
            action, _states = dqn_agent.model.predict(observation=obs, deterministic=True)
            o, r, te, tr, i = env.step(action)
            curr_tardiness.append(env.get_tardiness_percentage())
            tot_reward += r
            if te:
                break
        ep_reward.append(tot_reward)
        ep_tardiness.append(np.mean(curr_tardiness))
    return ep_reward, ep_tardiness


if __name__ == "__main__":
    from callback.plot_training_callback import PlotTrainingCallback
    MAX_STEPS = 300_000
    plot_training_callback: PlotTrainingCallback = PlotTrainingCallback(plot_freq=100)

    agent = Agent(custom_env=init_custom_factory_env(max_steps=MAX_STEPS))

    agent.learn(
        total_time_steps=MAX_STEPS, log_interval=10, callback=plot_training_callback
    )
    # # agent.learn()

    agent.save()

    # agent.load()
    # agent.evaluate()
