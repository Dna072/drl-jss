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
from stable_baselines3 import A2C
import numpy as np
import matplotlib.pyplot as plt
from custom_environment.utils import print_observation, print_jobs, print_scheduled_jobs, print_capacity_obs


class Agent:
    """
    DQN agent for learning the custom FactoryEnv environment
    """

    FILE_PATH_NAME: str = "files/a2c_custom_factory_env_multi_job"
    POLICY: str = (
        "MultiInputPolicy"  # converts multiple Dictionary inputs into a single vector
    )
    IS_VERBOSE: int = 1

    def __init__(self, custom_env: FactoryEnv | Monitor) -> None:
        self.custom_env: FactoryEnv = custom_env
        self.model: A2C = A2C(
            policy=self.POLICY, env=self.custom_env, verbose=self.IS_VERBOSE,
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
        self.model = A2C.load(path=file_path_name, env=self.custom_env)

    def evaluate(self,num_of_episodes: int = 10):
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
            # print_jobs(self.custom_env)
            action, _states = self.model.predict(observation=obs, deterministic=True)
            # print(f'Action: {action}')
            obs, reward, terminated, truncated, info = self.custom_env.step(
                action=action
            )
            # print_capacity_obs(obs, env=self.custom_env)
            # print_scheduled_jobs(self.custom_env, buffer_size=10)
            # print(f'Reward: {reward}, '
            #       f'Factory time: {info["CURRENT_TIME"]} '
            #       f'JOT: {info["JOBS_COMPLETED_ON_TIME"]}, '
            #       f'JNOT: {info["JOBS_NOT_COMPLETED_ON_TIME"]}, '
            #       f'UC_JOBS_BUFFER: {info["UNCOMPLETED_JOBS_BUFFER"]}, '
            #       f'LOST_JOBS: {info["LOST_JOBS"]}')
            #
            #
            # test = input('Enter to continue')

            returns.append(reward)
            curr_tardiness = self.custom_env.get_tardiness_percentage()
            jobs_ot = self.custom_env.get_jobs_completed_on_time()
            jobs_not = self.custom_env.get_jobs_completed_not_on_time()
            steps += 1

            if terminated or truncated:
                # print(f"avg return: { np.sum(returns) / steps}")
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


def episodic_a2c_agent(a2c_agent: Agent, n_episodes: int = 10):
    ep_reward = []
    ep_tardiness = []
    ep_jobs_ot = []
    ep_jobs_not = []
    #dqn_agent = Agent(custom_env=init_custom_factory_env(max_steps=env_max_steps, is_evaluation=True))
    #dqn_agent.load(agent_path)
    for e in range(n_episodes):
        env = a2c_agent.custom_env
        obs, info = env.reset()
        tot_reward = 0
        curr_tardiness = []
        while 1:  # the environment has its own termination clauses, so it will trigger the break
            action, _states = a2c_agent.model.predict(observation=obs, deterministic=True)
            o, r, te, tr, i = env.step(action)
            curr_tardiness.append(env.get_tardiness_percentage())
            tot_reward += r
            jobs_ot = env.get_jobs_completed_on_time()
            jobs_not = env.get_jobs_completed_not_on_time()
            if te:
                break
        ep_reward.append(tot_reward)
        ep_tardiness.append(np.mean(curr_tardiness))
        ep_jobs_ot.append(jobs_ot)
        ep_jobs_not.append(jobs_not)
    return ep_reward, ep_tardiness, ep_jobs_ot, ep_jobs_not


if __name__ == "__main__":
    from callback.plot_training_callback import PlotTrainingCallback
    LEARNING_MAX_STEPS = 20_100_000
    ENVIRONMENT_MAX_STEPS = 5_000
    JOBS_BUFFER_SIZE: int = 3
    N_MACHINES: int = 3
    N_RECIPES: int = 3
    plot_training_callback: PlotTrainingCallback = PlotTrainingCallback(plot_freq=10_000, algorithm="A2C")

    agent = Agent(custom_env=init_custom_factory_env(max_steps=ENVIRONMENT_MAX_STEPS, buffer_size=JOBS_BUFFER_SIZE,
                                              n_recipes=N_RECIPES, job_deadline_ratio=0.3, n_machines=N_MACHINES))

    #agent.load(file_path_name="files/trainedAgents/a2c_agent_seco_4_machines_24100000")
    agent.learn(
        total_time_steps=LEARNING_MAX_STEPS, log_interval=10, callback=plot_training_callback
    )
    agent.save(file_path_name=f"files/trainedAgents/a2c_seco_recipes_{N_MACHINES}m_{N_RECIPES}r_{JOBS_BUFFER_SIZE}b_real_"+str(LEARNING_MAX_STEPS))

    # agent.load(file_path_name="files/trainedAgents/a2c_agent_seco_4_machines_"+str(LEARNING_MAX_STEPS))
    # print(agent.evaluate())
