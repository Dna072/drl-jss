"""
Custom environment class for basic concept:
    1.  All jobs have only one recipe.
    2.  All jobs have a deadline.
    3.  All jobs are inserted into a machine using trays.
    4.  Machines can perform at least one recipe.
    5.  Machines have a maximum tray capacity as part of their attributes. Once a machine’s tray capacity is maxed out,
        no jobs can be assigned to it.
    6.  Recipes have a duration of time to complete.
    7.  The goal of the RL agent is to minimize tardiness (having most jobs completed before or on their deadline) and
        maximize efficiency of machines (machines should not be left idle for long periods).
    8.  The observation space is the job buffer (pending jobs to be scheduled), machines and their current capacity,
        and jobs completed.
    9.  The action of RL agent to select which job, J_i to assign to machine M_m, where 0 <= i < |J| and 0 <= m < |M|.
        The action space is thus all possible combinations of (J, M) with addition of No-Op action (taken at a timestep)

Base source:
    - https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html
"""

from custom_environment.machine import Machine
from custom_environment.job import Job

import gymnasium as gym
import numpy as np


class FactoryEnv(gym.Env):
    """
    Custom Environment that follows gym interface
    """

    NEUTRAL_REWARD = 0
    NEGATIVE_REWARD = -1

    NO_OP_SPACE = 1
    NO_OP_ACTION = -1

    metadata: dict = {"render_modes": ["vector"]}

    def __init__(
        self, machines: list[Machine], jobs: list[Job], max_steps: int = 10_000
    ) -> None:
        """
        FactoryEnv constructor method using gym.space objects for action and observation space

        machines: array of Machine (M) instances
        jobs: array of Job (J) instances
        pending_jobs: array of jobs pending for operation by a machine
        completed_jobs: array of jobs completed
        action_space: all possible combinations of (J, M) with addition of No-Op action (taken at a timestep)
        observation_space: dict containing arrays for pending jobs, machines and recipes and No-op, and completed jobs
        current_obs: the current observation dict taken at a time step, for when there is a no-op action
        max_steps: maximum steps for learning in the environment
        time_step: steps counter
        """
        super(FactoryEnv, self).__init__()

        self.machines: list[Machine] = machines
        self.jobs: list[Job] = jobs

        self.action_space: gym.spaces.Discrete = gym.spaces.Discrete(
            len(self.machines) * len(self.jobs) + self.NO_OP_SPACE
        )

        self.pending_jobs: list[Job] = self.jobs.copy()
        pending_jobs_space: gym.spaces.Box = gym.spaces.Box(
            low=0, high=1, shape=(len(self.jobs),), dtype=np.float32
        )

        machine_space: gym.spaces.Box = gym.spaces.Box(
            low=0,
            high=1,
            shape=(
                len(self.machines)
                * (
                    len(set().union(*[set(job.get_recipes()) for job in self.jobs]))
                    + self.NO_OP_SPACE
                ),
            ),
            dtype=np.float32,
        )

        self.completed_jobs: list[Job] = []
        jobs_completed_space: gym.spaces.Box = gym.spaces.Box(
            low=0, high=1, shape=(len(self.jobs),), dtype=np.float32
        )

        self.observation_space: gym.spaces.Dict = gym.spaces.Dict(
            {
                "pending_jobs": pending_jobs_space,
                "machines": machine_space,
                "completed_jobs": jobs_completed_space,
            }
        )
        self.current_obs: dict[str, np.ndarray[any]] = dict()

        self.max_steps: int = max_steps
        self.time_step: int = 0

    def get_obs(self) -> dict[str, np.ndarray[any]]:
        """
        return: observation dict containing binary arrays for pending jobs, machines' active jobs, and completed jobs
        """
        is_pending_jobs = np.zeros(len(self.jobs), dtype=np.float32)
        for job in self.pending_jobs:
            is_pending_jobs[job.get_job_id()] = 1.0

        is_machines_active_jobs = np.zeros(
            (len(self.machines), len(self.jobs) + self.NO_OP_SPACE), dtype=np.float32
        )
        for machine in self.machines:
            for job in machine.get_active_jobs():
                is_machines_active_jobs[
                    machine.get_machine_id(), job.get_job_id()
                ] = 1.0

        is_completed_jobs = np.zeros(len(self.jobs), dtype=np.uint8)
        for job in self.completed_jobs:
            is_completed_jobs[job.get_job_id()] = 1.0

        self.current_obs = {
            "pending_jobs": is_pending_jobs,
            "machines": is_machines_active_jobs.flatten(),
            "completed_jobs": is_completed_jobs,
        }
        return self.current_obs

    def get_reward(self, machine: Machine, job: Job) -> int:
        """
        Compute reward based on minimizing of tardiness and maximizing of machine efficiency
        """
        return 1  # TODO: implement method for computing reward based on tardiness and machine efficiency

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray[any]], int, bool, bool, dict[str, str]]:
        """
        Take a single step in the factory environment
        :param action: the agent's action to take in the step
        :return: (observation, reward, terminated, truncated, info)
        """
        self.time_step += 1
        is_terminated: bool = self.time_step > self.max_steps

        if action.item() == self.NO_OP_ACTION:
            return (
                self.current_obs,
                self.NEUTRAL_REWARD,
                is_terminated,
                False,
                {"Error": "No operation"},
            )

        job_id: int = action.item(0) // len(self.machines)
        if not 0 <= job_id < len(self.pending_jobs):
            return (
                self.get_obs(),
                self.NEGATIVE_REWARD,
                is_terminated,
                True,
                {"Error": "Invalid Job"},
            )
        job: Job = self.pending_jobs[job_id]

        machine_id: int = action.item(0) % len(self.machines)
        if self.machines[machine_id].assign_jobs(jobs=[job], recipe=job.get_recipes()):
            self.pending_jobs.remove(job)
            self.completed_jobs.append(job)
            return (
                self.get_obs(),
                self.get_reward(self.machines[machine_id], job),
                is_terminated,
                False,
                {},
            )
        return (
            self.get_obs(),
            self.NEGATIVE_REWARD,
            is_terminated,
            True,
            {"Error": "Invalid job assignment"},
        )

    def reset(
        self, seed: int = None, options: str = None
    ) -> tuple[dict[str, np.ndarray[any]], dict[str, str]]:
        self.time_step = 0

        for machine in self.machines:
            machine.reset()

        for job in self.jobs:
            job.reset()

        self.pending_jobs = self.jobs.copy()
        self.completed_jobs = []

        return self.get_obs(), {}

    def render(self):
        for machine in self.machines:
            print(machine)
            print("/-------------------------/")

    def close(self):
        self.reset()


def init_custom_factory_env(is_verbose: bool = False) -> FactoryEnv:
    """
    Create a custom FactoryEnv environment for development and testing
    :param is_verbose: print statements if True
    :return: custom FactoryEnv environment instance
    """
    machine_one: Machine = Machine(
        k_recipes=[1, 2], machine_id=0, m_type="A", cap=10_000
    )
    machine_two: Machine = Machine(
        k_recipes=[3, 2], machine_id=1, m_type="A", cap=10_000
    )
    machine_three: Machine = Machine(
        k_recipes=[2], machine_id=2, m_type="A", cap=10_000
    )

    if is_verbose:
        print(machine_one)
        print(machine_two)
        print(machine_three)
        print()

    job_one: Job = Job(
        recipes=["A1", "A2"],
        job_id=0,
        quantity=3,
        deadline="2024/01/04",
        priority=1,
    )
    job_two: Job = Job(
        recipes=["A2", "A3"],
        job_id=1,
        quantity=10,
        deadline="2024/02/04",
        priority=2,
    )
    job_three: Job = Job(
        recipes=["A3"], job_id=2, quantity=5, deadline="2023/12/04", priority=3
    )

    if is_verbose:
        job_one.recipe_in_progress("A1")
        job_one.recipe_completed("A1")

        print(job_one)
        print("/--------/")
        print(job_two)
        print("/--------/")
        print(job_three)

        job_one.reset()

    factory_env: FactoryEnv = FactoryEnv(
        machines=[machine_one, machine_two, machine_three],
        jobs=[job_one, job_two, job_three],
    )
    return factory_env


if __name__ == "__main__":
    from stable_baselines3.common.env_checker import check_env

    custom_factory_env = init_custom_factory_env(is_verbose=True)
    print("\nCustom environment check errors:", check_env(custom_factory_env))
