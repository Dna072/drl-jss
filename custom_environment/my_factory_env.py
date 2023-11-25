import gymnasium as gym
import numpy as np
from environment import FactoryEnv
from job import Job
from machine import Machine

class FacEnv2(FactoryEnv):
    def __init__(self, machines: list[Machine], jobs: list[Job], max_steps: int = 10000) -> None:
        super().__init__(machines, jobs, max_steps)

        self.job_remaining_times_space: gym.spaces.Box = gym.spaces.Box(
            low=0, high=np.inf, shape=(self.__BUFFER_LEN,), dtype=np.float64
        )

        self.observation_space: gym.spaces.Dict = gym.spaces.Dict(
            {
                self.__PENDING_JOBS_STR: self.pending_jobs_space,
                self.__MACHINES_STR: self.machine_space,
                self.__JOB_REMAINING_TIMES_STR: self.job_remaining_times_space,
                self.__ACHIEVED_GOAL_STR: self.achieved_goal_space,
                self.__DESIRED_GOAL_STR: self.desired_goal_space
            }
        )

if __name__ == "__main__":
    factory = FacEnv2(machines=[], jobs=[])
    print("Working...")