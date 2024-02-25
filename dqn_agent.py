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
from custom_environment.utils import (print_observation,
                                      print_jobs, print_scheduled_jobs,
                                      print_uncompleted_jobs_buffer, print_capacity_obs)


class Agent:
    """
    DQN agent for learning the custom FactoryEnv environment
    """

    FILE_PATH_NAME: str = "files/dqn_custom_factory_env_multi_recipe_3"
    POLICY: str = (
        "MultiInputPolicy"  # converts multiple Dictionary inputs into a single vector
    )
    IS_VERBOSE: int = 1

    def __init__(self, custom_env: FactoryEnv | Monitor) -> None:
        self.custom_env: FactoryEnv = custom_env
        self.model: DQN = DQN(
            policy=self.POLICY, env=self.custom_env, verbose=self.IS_VERBOSE,
             gamma=0.9, exploration_fraction=0.5, batch_size=10_000
            # learning_rate=1e-3, gamma=0.6, exploration_fraction=0.25,buffer_size=10_000
        )

    def learn(
        self,
        total_time_steps: int = 100_000,
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

    def evaluate(self, num_of_episodes: int = 10):
        obs, info = self.custom_env.reset()
        avg_returns_per_episode = []
        returns = []
        ep_reward = []
        ep_tardiness = []
        ep_jobs_ot = []
        ep_jobs_not = []
        episode = 0
        steps = 0

        while episode < num_of_episodes:
            print_jobs(self.custom_env)
            print_uncompleted_jobs_buffer(self.custom_env)
            print_capacity_obs(obs, machines=self.custom_env.get_machines(), n_machines=2, print_length=10)


            action, _states = self.model.predict(observation=obs, deterministic=True)
            print(f'Action: {action}')
            obs, reward, terminated, truncated, info = self.custom_env.step(
                action=action
            )
            print_scheduled_jobs(self.custom_env)
            print(f'Reward: {reward}, '
                  f'Factory time: {info["CURRENT_TIME"]} '
                  f'JOT: {info["JOBS_COMPLETED_ON_TIME"]}, '
                  f'JNOT: {info["JOBS_NOT_COMPLETED_ON_TIME"]}, '
                  f'UC_JOBS_BUFFER: {info["UNCOMPLETED_JOBS_BUFFER"]}, '
                  f'LOST_JOBS: {info["LOST_JOBS"]}')


            test = input('Enter to continue')

            returns.append(reward)
            curr_tardiness = self.custom_env.get_tardiness_percentage()
            jobs_ot = self.custom_env.get_jobs_completed_on_time()
            jobs_not = self.custom_env.get_jobs_completed_not_on_time()
            steps += 1

            if terminated or truncated:
                # print(f"avg return: {np.sum(returns) / steps}")
                avg_returns_per_episode.append(np.sum(returns) / steps)
                ep_reward.append(np.sum(returns))
                ep_tardiness.append(curr_tardiness)
                ep_jobs_ot.append(jobs_ot)
                ep_jobs_not.append(jobs_not)
                steps = 0
                returns = []
                obs, info = self.custom_env.reset()
                episode += 1
        return ep_reward, ep_tardiness, ep_jobs_ot, ep_jobs_not


def episodic_dqn_agent(n_episodes: int = 10, agent_path: str = "files/dqn_custom_factory_env_multi_recipe_2",
                       env_max_steps: int = 100):
    ep_reward = []
    ep_tardiness = []
    ep_jobs_ot = []
    ep_jobs_not = []
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
            jobs_ot = env.get_jobs_completed_on_time()
            jobs_not = env.get_jobs_completed_not_on_time()
            tot_reward += r
            if te:
                break
        ep_reward.append(tot_reward)
        ep_tardiness.append(np.mean(curr_tardiness))
        ep_jobs_ot.append(jobs_ot)
        ep_jobs_not.append(jobs_not)
    return ep_reward, ep_tardiness, ep_jobs_ot, ep_jobs_not


if __name__ == "__main__":
    from callback.plot_training_callback import PlotTrainingCallback
    LEARNING_MAX_STEPS = 8_100_000
    ENVIRONMENT_MAX_STEPS = 50_000
    plot_training_callback: PlotTrainingCallback = PlotTrainingCallback(plot_freq=10_000)

    agent = Agent(custom_env=init_custom_factory_env(max_steps=ENVIRONMENT_MAX_STEPS))

    # agent.learn(
    #     total_time_steps=LEARNING_MAX_STEPS, log_interval=1000, callback=plot_training_callback
    # )
    # agent.save(file_path_name="files/trainedAgents/dqn_agent_multi_recipe_job_"+str(LEARNING_MAX_STEPS))

    agent.load(file_path_name='files/trainedAgents/dqn_agent_multi_recipe_job_8100000')
    agent.evaluate(num_of_episodes = 1_000)
