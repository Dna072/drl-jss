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

from itertools import permutations
import gymnasium as gym
import numpy as np
import datetime


class FactoryEnv(gym.Env):
    """
    Custom Environment that follows gym interface
    """

    NEUTRAL_REWARD = 0
    NEGATIVE_REWARD = -1

    NO_OP_SPACE = 1
    NO_OP_ACTION = -1

    REWARD_WEIGHTS = {"Idle": -1, "Deadline": -5}

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
            2 ** (len(self.machines) * (len(self.jobs) + self.NO_OP_SPACE))
        )

        self.pending_jobs: list[Job] = self.jobs.copy()
        pending_jobs_space: gym.spaces.Box = gym.spaces.Box(
            low=0, high=1, shape=(len(self.jobs),), dtype=np.float64
        )

        machine_space: gym.spaces.Box = gym.spaces.Box(
            low=0,
            high=1,
            shape=(len(self.machines) * (len(self.jobs) + self.NO_OP_SPACE),),
            dtype=np.float64,
        )

        self.completed_jobs: list[Job] = []
        jobs_completed_space: gym.spaces.Box = gym.spaces.Box(
            low=0, high=1, shape=(len(self.jobs),), dtype=np.float64
        )

        # the achieved_goal and desired_goal are required for a "goal-conditioned env"
        self.achieved_goal_space: list[int] = [0]
        achieved_goal_space: gym.spaces.Box = gym.spaces.Box(
            low=0, high=1, shape=(len(self.achieved_goal_space),), dtype=np.float64
        )
        self.desired_goal_space: list[int] = [1]
        desired_goal_space: gym.spaces.Box = gym.spaces.Box(
            low=0, high=1, shape=(len(self.desired_goal_space),), dtype=np.float64
        )

        self.observation_space: gym.spaces.Dict = gym.spaces.Dict(
            {
                "pending_jobs": pending_jobs_space,
                "machines": machine_space,
                "completed_jobs": jobs_completed_space,
                "achieved_goal": achieved_goal_space,
                "desired_goal": desired_goal_space,
            }
        )
        self.current_obs: dict[str, np.ndarray[any]] = dict()

        self.max_steps: int = max_steps
        self.time_step: int = 0

        self.episode_reward_sum: int = 0

    def get_legal_actions(self) -> list[list[int]]:
        """
        Function to return the legal actions in the environment.
        This assumes you can only assign jobs to one machine at a time step not considering machine capacity constraints
        """
        r: int = len(self.jobs)

        actions: list[list[int]] = []
        all_actions: list[any]

        for m in self.machines:
            # job_binary_string = ''
            all_actions = list(set(["".join(p) for p in permutations("01" * r, r)]))

            for idx, j in enumerate(self.jobs):
                # Assuming each job has only one recipe, otherwise, we have to use another approach on this
                for recipe in j.recipes:
                    if not m.can_perform_recipe(recipe):
                        # filter through actions and select actions with this index set to 0
                        print(
                            f"Machine {m.machine_id} cannot perform {recipe} idx {idx}"
                        )
                        all_actions = list(
                            filter(lambda action: action[idx] == "0", all_actions)
                        )

            print(all_actions)
            actions.append(
                [int(x, 2) for x in all_actions]
            )  # Convert actions from binary string to integer

        return actions

    def get_obs(self) -> dict[str, np.ndarray[float]]:
        """
        return: observation dict containing binary arrays for pending jobs, machines' active jobs, and completed jobs
        """
        is_pending_jobs: np.ndarray[float] = np.zeros(len(self.jobs), dtype=np.float64)
        for job in self.pending_jobs:
            is_pending_jobs[job.get_job_id()] = 1.0

        is_machines_active_jobs: np.ndarray[np.ndarray[float]] = np.zeros(
            (len(self.machines), (len(self.jobs) + self.NO_OP_SPACE)), dtype=np.float64
        )
        for machine in self.machines:
            # print("AJ:",machine.get_active_jobs())
            for job in machine.get_active_jobs():
                is_machines_active_jobs[
                    machine.get_machine_id(), job.get_job_id()
                ] = 1.0

        is_completed_jobs: np.ndarray[float] = np.zeros(
            len(self.jobs), dtype=np.float64
        )
        for job in self.completed_jobs:
            is_completed_jobs[job.get_job_id()] = 1.0

        self.achieved_goal_space = [0]  # TODO: compute this value for each step
        achieved_goals_space: np.ndarray[float] = np.zeros(
            len(self.achieved_goal_space), dtype=np.float64
        )
        achieved_goals_space[0] = self.achieved_goal_space[0]

        self.desired_goal_space = [1]  # TODO: compute this value for each step
        desired_goal_space: np.ndarray[float] = np.zeros(
            len(self.desired_goal_space), dtype=np.float64
        )
        desired_goal_space[0] = self.desired_goal_space[0]

        self.current_obs: dict[str, np.ndarray[float]] = {
            "pending_jobs": is_pending_jobs,
            "machines": is_machines_active_jobs.flatten(),
            "completed_jobs": is_completed_jobs,
            "achieved_goal": achieved_goals_space,  # required since is now a "goal-conditioned env" obs
            "desired_goal": desired_goal_space,  # required since is now a "goal-conditioned env" obs
        }
        return self.current_obs

    def translate_action_to_jobs(self, action) -> list[tuple[str, list[int]]] | None:
        max_len: int = len(self.machines) * (
            len(self.jobs) + 1
        )  # to fill with zeros to the left
        # print(max_len)

        try:
            bin_act: str = bin(action)
            machine_jobs_tuples: list[tuple[str, list[int]]] = []
            # binary_length = len(bin_act)
            len(bin_act)

            # split the action in machine + actions
            m: int = len(self.machines)
            bin_act_str: str = str(bin_act.replace("0b", "")).zfill(max_len)

            # print(bin_act_str)
            part_length: int = len(bin_act_str) // m
            machines_binary: list[str] = [
                bin_act_str[i * part_length : (i + 1) * part_length] for i in range(m)
            ]
            for i, machine_binary in enumerate(machines_binary):
                selected_jobs: list[any] = []

                for j, bit in enumerate(machine_binary):
                    if bit == "1":
                        selected_jobs.append(part_length - j - 1)
                machine_jobs_tuples.append(
                    (i, selected_jobs)
                )  # (machine, selected_jobs)
            return machine_jobs_tuples
        except ValueError:
            return None

    def compute_custom_reward(self) -> int:
        """
        Compute reward based on minimizing of tardiness and maximizing of machine efficiency
        REWARD_WEIGHTS = {"Idle": -1, "Deadline": -5, "Completed": 4}
        Note: there is a parameter conflict when using the method name: compute_reward
        """
        total_time_unused: int = 0
        total_time_deadlines: int = 0
        custom_reward: int

        for m in self.machines:
            # compute free time
            if m.get_status() == 0:
                total_time_unused += (
                    (datetime.datetime.now() - m.get_timestamp_status()).seconds
                ) // 3600  # round - in hours

        time_past: int
        for j in self.jobs:
            # calculate partial weights proportional to deadline. 40% if < 10 hours, 80% < 24hours, 100% >24 hours
            if j.get_deadline_datetime() < datetime.datetime.now():
                time_past = (
                    (datetime.datetime.now() - j.get_deadline_datetime()).seconds
                ) // 3600

                if time_past < 10:
                    total_time_deadlines += 0.4 * time_past
                elif 10 <= time_past < 24:
                    total_time_deadlines += 0.8 * time_past
                else:
                    total_time_deadlines += time_past
        print("Values:\nTTU:", total_time_unused, "\nTTD:", total_time_deadlines)

        custom_reward = int(
            self.REWARD_WEIGHTS["Idle"] * total_time_unused
            + self.REWARD_WEIGHTS["Deadline"] * total_time_deadlines
        )
        self.episode_reward_sum += custom_reward
        return custom_reward

    def get_jobs(self, ids) -> list[Job]:
        requested: list[Job] = []
        for j in self.jobs:
            if j.get_job_id() in ids:
                requested.append(j)
        return requested

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray[any]], int, bool, bool, dict[str, str]]:
        """
        Take a single step in the factory environment
        :param action: the agent's action to take in the step
        :return: (observation, reward, terminated, truncated, info)
        """
        self.time_step += 1
        # TODO: maybe check time and update any necessary machines if their time to complete has finished, on every step

        is_terminated: bool = self.time_step > self.max_steps

        # action_binary: str = bin(action)

        # Jobs to do based on action
        to_do: list[tuple[str, list[int]]] = self.translate_action_to_jobs(action)

        # # first get valid actions or validate the action after?
        # if action.item() == self.NO_OP_ACTION:
        #     return (
        #         self.current_obs,
        #         self.NEUTRAL_REWARD,
        #         is_terminated,
        #         False,
        #         {"Error": "No operation"},
        #     )

        # job_id: float = action.item(0) // len(self.machines)

        # print(to_do)
        if to_do is None:
            return (
                self.get_obs(),  # observation
                self.NEGATIVE_REWARD,  # reward
                is_terminated,  # terminated
                True,  # truncated
                {"Error": "Invalid Action"},  # info
            )

        # job: Job = self.pending_jobs[job_id]
        # machine_id: int = action.item(0) % len(self.machines)

        try:
            for t in to_do:
                m: str = t[0]
                js: list[int] = t[1]

                # print("Machine:", m ,"jobs:", js)
                if js:
                    # print(self.machines[m].get_machine_type())
                    # print(self.get_jobs(js))
                    if self.machines[int(m)].assign_jobs(jobs=self.get_jobs(js)):
                        ####################################
                        # TODO: What to do with pending jobs
                        ####################################
                        for _ in self.pending_jobs:
                            print("")

            # if pj.get_job_id() in js:
            #     self.pending_jobs.remove(self.get_jobs(pj))
            #     self.completed_jobs.append(job)

            return (
                self.get_obs(),
                self.compute_custom_reward(),  # TODO: fix parameter conflict when using compute_reward
                is_terminated,
                False,
                {},
            )
        except Exception:
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
        """
        Reset environment
        """
        self.time_step = 0
        self.episode_reward_sum = 0

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

    def close(self) -> None:
        self.reset()
