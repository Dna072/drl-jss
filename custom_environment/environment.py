"""
Custom FactoryEnv class for basic concept:
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

Base source:
    - https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html
"""
import math

# import math

from custom_environment.machine import Machine
from custom_environment.job import Job
from custom_environment.recipe import Recipe
from custom_environment.recipe_factory import create_recipe
from custom_environment.job_factory import create_job
from custom_environment.utils import min_max_norm, min_max_norm_list

# from datetime import datetime, timedelta
import gymnasium as gym

# from time import sleep
import numpy as np
import random


class TextColors:
    RESET = "\033[0m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"


class FactoryEnv(gym.Env):
    """
    Custom Environment that follows gym interface
    """

    ####################
    # public constants #
    ####################

    NO_OP_STR: str = "NO-OP"
    MACHINE_IDLE_STR: str = "MACHINE_IDLE"
    MACHINE_UNAVAILABLE_STR: str = "MACHINE UNAVAILABLE"
    JOB_ASSIGNED_STR: str = "JOB_ASSIGNED"
    JOB_COMPLETED_ON_TIME_STR: str = "JOB_COMPLETED_ON_TIME"
    JOB_COMPLETED_NOT_ON_TIME_STR: str = "JOB_COMPLETED_NOT_ON_TIME"
    INVALID_JOB_RECIPE_STR: str = "INVALID JOB RECIPE"
    DEADLINE_EXCEEDED_STR: str = "DEADLINE_EXCEEDED"
    ILLEGAL_ACTION_STR: str = "ILLEGAL ACTION"
    MACHINE_UNDER_UTILIZED_STR: str = "MACHINE_UNDER_UTILIZED"
    NEUTRAL_STR: str = "NEUTRAL"

    MAX_RECIPES_IN_ENV_SYSTEM: int = 1
    r1 = create_recipe(
        factory_id="R1_ID", process_time=30.0, process_id=0, recipe_type="R1"
    )
    r2 = create_recipe(
        factory_id="R2_ID", process_time=300.0, process_id=1, recipe_type="R2"
    )
    # available_recipes = [r1, r2]

    #####################
    # private constants #
    #####################
    _JOBS_QUEUE_LEN: int = 10
    _BUFFER_LEN: int = 5
    _MAX_NEXT_RECIPES: int = 1 #variable to control the number of recipes for a job the agent can see to help in forecasting
    _NO_OP_SPACE: int = 1
    _START_MACHINES_SPACE: int = 1
    _MAX_MACHINES: int = 2
    _REWARD_WEIGHTS: dict[str, int] = {
        NO_OP_STR: 0,
        MACHINE_IDLE_STR: -1,
        MACHINE_UNAVAILABLE_STR: -20,
        JOB_COMPLETED_ON_TIME_STR: 10,
        JOB_COMPLETED_NOT_ON_TIME_STR: -15,
        JOB_ASSIGNED_STR: 5,
        INVALID_JOB_RECIPE_STR: -30,
        DEADLINE_EXCEEDED_STR: -5,
        ILLEGAL_ACTION_STR: -15,
        MACHINE_UNDER_UTILIZED_STR: -20,
        NEUTRAL_STR: 0,
    }

    _REWARD_TIME_PENALTIES: dict[str, dict[str, int | float]] = {
        "10_hrs": {"in_s": 36_000, "weight": 0.4},
        "24_hrs": {"in_s": 86_400, "weight": 0.8},
    }

    # observation space constant dict keys
    _NEW_JOBS_QUEUE_STR: str = "new_jobs_queue"
    _NEW_JOBS_TRAY_CAPACITIES_STR: str = "new_jobs_tray_capacities"
    _PENDING_JOBS_STR: str = "pending_jobs"
    _RECIPE_TYPES_STR: str = "recipes"
    _MACHINES_STR: str = "machines"
    _MACHINES_PENDING_CAPACITY_STR: str = "machine_pending_capacity"
    _MACHINES_ACTIVE_CAPACITY_STR: str = "machine_active_capacity"
    _MACHINES_ACTIVE_RECIPE_STR: str = "machine_active_recipe"
    _MACHINE_RECIPES: str = "machine_recipes"
    _MACHINE_AVAILABLE_CAPACITY_STR: str = "machine_available_capacity"
    _P_JOB_RECIPE_STR: str = "pending_job_recipe"
    _P_JOB_RECIPE_COUNT_STR: str = "pending_job_recipe_count"
    _P_JOB_REMAINING_TIMES_STR: str = "pending_job_remaining_times"
    _P_JOB_PROCESS_TIME_TO_DEADLINE_RATIO: str = "pending_job_process_time_deadline_ratio"
    _P_JOB_STEPS_TO_DEADLINE: str = "pending_job_steps_to_deadline"
    _P_JOB_NEXT_RECIPES: str = "pending_job_next_recipes"
    _P_JOB_TRAY_CAPACITIES: str = "pending_job_tray_capacities"
    _UC_JOB_NEXT_RECIPES: str = "uncompleted_job_buffer_next_recipes"
    _UC_JOB_RECIPE_STR: str = "uncompleted_job_recipes"
    _UC_JOB_RECIPE_COUNT_STR: str = "uncompleted_job_recipe_count"
    _UC_JOB_BUFFER_RECIPES: str = "uncompleted_job_buffer_recipes"
    _UC_JOB_BUFFER_RECIPE_COUNT: str = "uncompleted_job_buffer_recipe_count"
    _UC_JOB_REMAINING_TIMES: str = "uncompleted_job_remaining_times"
    _UC_BUFFER_PROCESS_TIME_TO_DEADLINE_RATIO: str = "uncompleted_job_buffer_process_time_deadline_ratio"
    _UC_JOB_BUFFER_REMAINING_TIMES: str = "uncompleted_job_remaining_times"
    _LOST_JOBS_COUNT: str = "lost_jobs_count"

    _METADATA: dict[int, str] = {0: "vector", 1: "human"}

    def __init__(
        self, machines: list[Machine], jobs: list[Job], recipes: list[Recipe], recipe_probs: list[float],
            max_steps: int = 10_000, is_evaluation: bool = False, jobs_buffer_size: int = 3, jobs_queue_size: int = 10,
            job_deadline_ratio: float = 0.3, n_machines: int = 2, machine_tray_capacity: int = 40, refresh_arrival_time: bool = False
    ) -> None:
        """
        FactoryEnv class constructor method using gym.Space objects for action and observation space
        :param machines: array of Machine instances
        :param jobs: array of Job instances
        :param max_steps: maximum episode length
        :param refresh_arrival_time: Update the arrival time to now for jobs coming from the queue once it is inserted into the buffer
        :
        """
        super(FactoryEnv, self).__init__()
        self._total_machines_available: list[Machine] = machines
        self._MAX_MACHINES = n_machines
        self._MACHINE_TRAY_CAPACITY = machine_tray_capacity
        self._machines: list[Machine] = self._total_machines_available.copy()[
            : self._MAX_MACHINES
        ]  # restricted len for max machines being used
        self.is_evaluation: bool = is_evaluation
        self._BUFFER_LEN = jobs_buffer_size
        self._JOBS_QUEUE_LEN = jobs_queue_size
        self._jobs_queue: list[Job] = jobs.copy()
        self.available_recipes = recipes
        self.available_recipe_probs = recipe_probs
        self.job_deadline_ratio = job_deadline_ratio
        self._pending_jobs: list[Job] = jobs.copy()[
            : self._BUFFER_LEN
        ]  # buffer of restricted len
        self._fresh_arrival_time = refresh_arrival_time
        self._jobs_in_progress: list[tuple[Machine, Job]] = []
        self._completed_jobs: list[Job] = []
        self._uncompleted_jobs: list[Job] = []
        self._uncompleted_jobs_buffer: list[Job] = [] #uncompleted jobs schedulable to machines, length = BUFFER_LEN
        self._lost_jobs: list[Job] = [] # jobs that may never be completed since they could not be in the uncompleted jobs buffer
        self._jobs_completed_per_step_on_time: int = 0
        self._jobs_completed_per_step_not_on_time: int = 0

        # variable to keep track of time past deadline for jobs completed after their deadline
        self._late_jobs_time_past_deadline: list[float] = []

        self._max_steps: int = max_steps
        self._time_step: int = 0
        self._step_datetime: str | None = None

        self.factory_time: int = (
            0  # NOTE: This is the variable that tracks processing time
        )
        self._termination_reward: float = -6000  # NOTE: Check if applicable
        ############
        # callback #
        ############

        self.episode_reward_sum: float = 0.0  # for callback graphing train performance
        self.callback_step_reward: float = (
            0.0  # for callback graphing train performance
        )
        self.callback_step_tardiness: float = 0.0
        self.callback_flag_termination: bool = (
            False  # for callback graphing train performance
        )

        ################
        # action space #
        ################

        self.action_space = gym.spaces.Discrete(
            len(self._machines) * self._BUFFER_LEN + self._NO_OP_SPACE + len(self._machines)
        )# space multiplied by 2 to cater for actions from uncompleted_jobs_buffer

        #####################
        # observation space #
        #####################
        new_jobs_recipe_space: gym.spaces.Box = gym.spaces.Box(
            low=0, high=1, shape=(self._JOBS_QUEUE_LEN,), dtype=np.float64
        )
        new_jobs_tray_capacities_space: gym.spaces.Box = gym.spaces.Box(
            low=0, high=1, shape=(self._JOBS_QUEUE_LEN,), dtype=np.float64
        )
        pending_jobs_recipe_space: gym.spaces.Box = gym.spaces.Box(
            low=0, high=1, shape=(self._BUFFER_LEN,), dtype=np.float64
        )

        pending_jobs_tray_capacities: gym.spaces.Box = gym.spaces.Box(
            low=0, high=1, shape=(self._BUFFER_LEN, ), dtype=np.float64
        )
        pending_job_remaining_times_space: gym.spaces.Box = gym.spaces.Box(
            low=0, high=1, shape=(self._BUFFER_LEN,), dtype=np.float64
        )  # normalized vector for jobs pending deadline proportional to recipe processing duration times

        pending_job_process_time_deadline_ratio: gym.spaces.Box = gym.spaces.Box(
            low=0, high=1, shape=(self._BUFFER_LEN,), dtype=np.float64
        )
        pending_job_recipe_count_space: gym.spaces.Box = gym.spaces.Box(
            low=1, high=self.MAX_RECIPES_IN_ENV_SYSTEM, shape=(self._BUFFER_LEN,), dtype=np.float64
        )

        pending_job_next_recipes: gym.spaces.Box = gym.spaces.Box(
            low=-1, high=len(self.available_recipes) - 1, shape=(self._BUFFER_LEN * self._MAX_NEXT_RECIPES,), dtype=np.float64
        )

        machine_pending_capacity_space: gym.spaces.Box = gym.spaces.Box(
            low=0, high=1, shape=(len(self._machines),), dtype=np.float64
        ) # normalized vector for utilized machine tray capacity for scheduled jobs

        machine_active_capacity_space: gym.spaces.Box = gym.spaces.Box(
            low=0, high=1, shape=(len(self._machines),), dtype=np.float64
        )# normalized vector for utilized machine tray capacity for active jobs

        machine_active_recipe_space: gym.spaces.Box = gym.spaces.Box(
            low=-1, high=1, shape=(len(self._machines),), dtype=np.float64
        )

        machine_recipes_space: gym.spaces.Box = gym.spaces.Box(
            low=0, high=1, shape=(len(self.available_recipes) * len(self._machines),), dtype=np.float64
        )

        uncompleted_job_buffer_next_recipes: gym.spaces.Box = gym.spaces.Box(
            low=-1, high=len(self.available_recipes) - 1, shape=(self._BUFFER_LEN * self._MAX_NEXT_RECIPES,),
            dtype=np.float64
        )
        uncompleted_job_buffer_recipe_space: gym.spaces.Box = gym.spaces.Box(
            low=-1, high=len(self.available_recipes) - 1, shape=(self._BUFFER_LEN,), dtype=np.float64
        )

        uncompleted_job_buffer_remaining_times: gym.spaces.Box = gym.spaces.Box(
            low=-1, high=1, shape=(self._BUFFER_LEN,), dtype=np.float64
        )

        uncompleted_job_buffer_recipe_count_space: gym.spaces.Box = gym.spaces.Box(
            low=0, high=self.MAX_RECIPES_IN_ENV_SYSTEM, shape=(self._BUFFER_LEN,), dtype=np.float64
        )

        uncompleted_job_buffer_process_time_deadline_ratio: gym.spaces.Box = gym.spaces.Box(
            low=0, high=1, shape=(self._BUFFER_LEN,), dtype=np.float64
        )

        lost_jobs_count_space: gym.spaces.Box = gym.spaces.Box(
            low=0, high=2**8, shape=(1,), dtype=np.float64
        ) # keeps track of the number of lost jobs in the env

        pending_job_steps_to_deadline_space: gym.spaces.Box = gym.spaces.Box(
            low=0, high=1, shape=(self._BUFFER_LEN, ), dtype=np.float64
        )  # normalized vector (using Min-max scaling [0,1]) for steps to deadline for jobs in buffer

        self.observation_space: gym.spaces.Dict = gym.spaces.Dict(
            {
                # self._PENDING_JOBS_STR: pending_jobs_space,
                #self._MACHINES_STR: machine_space,
                self._NEW_JOBS_QUEUE_STR: new_jobs_recipe_space,
                self._NEW_JOBS_TRAY_CAPACITIES_STR: new_jobs_tray_capacities_space,
                self._P_JOB_RECIPE_STR: pending_jobs_recipe_space,
                self._MACHINES_PENDING_CAPACITY_STR: machine_pending_capacity_space,
                self._MACHINES_ACTIVE_CAPACITY_STR: machine_active_capacity_space,
                self._MACHINES_ACTIVE_RECIPE_STR: machine_active_recipe_space,
                self._MACHINE_RECIPES: machine_recipes_space,
                #self._P_JOB_REMAINING_TIMES_STR: pending_job_remaining_times_space,
                self._P_JOB_PROCESS_TIME_TO_DEADLINE_RATIO: pending_job_process_time_deadline_ratio,
                self._P_JOB_STEPS_TO_DEADLINE: pending_job_steps_to_deadline_space,
                self._P_JOB_TRAY_CAPACITIES: pending_jobs_tray_capacities
                # self._P_JOB_NEXT_RECIPES: pending_job_next_recipes,
                # # self._UC_JOB_RECIPE_STR: uncompleted_job_recipe_space,
                # self._UC_JOB_BUFFER_RECIPES: uncompleted_job_buffer_recipe_space,
                # # self._UC_JOB_REMAINING_TIMES: uncompleted_job_remaining_times,
                # self._UC_JOB_BUFFER_REMAINING_TIMES: uncompleted_job_buffer_remaining_times,
                # self._UC_BUFFER_PROCESS_TIME_TO_DEADLINE_RATIO: uncompleted_job_buffer_process_time_deadline_ratio,
                # self._P_JOB_RECIPE_COUNT_STR: pending_job_recipe_count_space,
                # # self._UC_JOB_RECIPE_COUNT_STR: uncompleted_job_recipe_count_space,
                # self._UC_JOB_BUFFER_RECIPE_COUNT: uncompleted_job_buffer_recipe_count_space,
                # self._UC_JOB_NEXT_RECIPES: uncompleted_job_buffer_next_recipes,
                # self._LOST_JOBS_COUNT: lost_jobs_count_space
                # self._P_JOB_STEPS_TO_DEADLINE: pending_job_steps_to_deadline_space
            }
        )

    def get_obs(self) -> dict[str, np.ndarray[float]]:
        """
        return: obs dict containing binary arrays for pending jobs, machine activity, and pending deadline durations
        """
        ###################################
        # update pending jobs observation #
        ###################################
        is_pending_jobs: np.ndarray = np.zeros(self._BUFFER_LEN, dtype=np.float64)
        for job in self._pending_jobs:
            is_pending_jobs[job.get_id()] = 1.0

        ###############################################################
        # update mapping jobs to machines processing them observation #
        ###############################################################
        is_machines_active_jobs: np.ndarray = np.zeros(
            (len(self._machines), self._BUFFER_LEN + 1), dtype=np.float64
        )
        for machine in self._machines:
            is_machines_active_jobs[machine.get_id(), self._BUFFER_LEN] = machine.get_pending_tray_capacity()

            for job in machine.get_active_jobs():
                is_machines_active_jobs[machine.get_id(), job.get_id()] = 1.0

        #######################################################################################################
        # update incomplete job pending deadline proportional to recipe processing duration times observation #
        #######################################################################################################
        max_duration = min_duration = 0
        pending_job_remaining_times: np.ndarray = np.zeros(
            self._BUFFER_LEN, dtype=np.float64
        )

        for job in self._pending_jobs:
            pending_job_remaining_times[job.get_id()] = job.get_remaining_process_time()
            # update max and min duration times for normalizing [0, 1]
            if pending_job_remaining_times[job.get_id()] > max_duration:
                max_duration = pending_job_remaining_times[job.get_id()]
            elif pending_job_remaining_times[job.get_id()] < min_duration:
                min_duration = pending_job_remaining_times[job.get_id()]
        if not sum(pending_job_remaining_times) == 0:
            for job in self._pending_jobs:
                pending_job_remaining_times[job.get_id()] = (
                    pending_job_remaining_times[job.get_id()] - min_duration
                ) / (max_duration - min_duration)

        pending_jobs_steps_to_deadline = [job.get_steps_to_deadline() for job in self._pending_jobs]
        min_deadline = min(pending_jobs_steps_to_deadline)
        max_deadline = max(pending_jobs_steps_to_deadline)
        pending_jobs_steps_to_deadline = np.array([min_max_norm(x, min_deadline, max_deadline) for x in pending_jobs_steps_to_deadline], dtype=np.float64)

        for idx, val in enumerate(pending_job_remaining_times):
            pending_job_remaining_times[idx] = pending_job_remaining_times[idx] * pending_jobs_steps_to_deadline[idx]

        ###########################################################
        # return current observation state object for step update #
        ###########################################################
        return {
            # self._PENDING_JOBS_STR: is_pending_jobs,
            self._MACHINES_STR: is_machines_active_jobs.flatten(),
            self._P_JOB_REMAINING_TIMES_STR: pending_job_remaining_times,
        }

    def get_capacity_obs(self) -> dict[str, np.ndarray[float]]:
        """
        return: obs dict containing binary arrays for pending jobs, machine activity, and pending deadline durations
        """
        ###################################
        # update pending jobs observation #
        ###################################
        is_pending_jobs: np.ndarray = np.zeros(self._BUFFER_LEN, dtype=np.float64)
        for idx, job in enumerate(self._pending_jobs):
            is_pending_jobs[idx] = 1.0

        ###############################################################
        # update mapping jobs to machines processing them observation #
        ###############################################################
        machine_pending_capacity_utilization: np.ndarray = np.zeros(len(self._machines), dtype=np.float64)
        machine_active_capacity_utilization: np.ndarray = np.zeros(len(self._machines), dtype=np.float64)

        machine_active_recipe: np.ndarray = np.full(
            len(self._machines), fill_value=-1, dtype=np.float64
        )
        machine_recipes: np.ndarray = np.full(
            (len(self._machines), len(self.available_recipes)), fill_value=-1, dtype=np.float64
        )
        # is_machines_active_jobs: np.ndarray = np.zeros(
        #     (len(self._machines), self._BUFFER_LEN), dtype=np.float64
        # )
        for machine in self._machines:
            machine_pending_capacity_utilization[machine.get_id()] = (machine.get_tray_capacity() - machine.get_pending_tray_capacity()) / machine.get_tray_capacity()
            machine_active_capacity_utilization[machine.get_id()] = (machine.get_tray_capacity() - machine.get_active_tray_capacity()) / machine.get_tray_capacity()
            if len(machine.get_pending_jobs()) > 0:
                machine_active_recipe[machine.get_id()] = min_max_norm(machine.get_active_recipe().get_id(), 0, len(self.available_recipes) - 1)

            for idx, recipe in enumerate(self.available_recipes):
                if recipe.get_factory_id() in machine.get_valid_recipes():
                    machine_recipes[machine.get_id(), idx] = min_max_norm(recipe.get_id(), 0, len(self.available_recipes) - 1)
            # for job in machine.get_pending_jobs():
            #     is_machines_active_jobs[machine.get_id(), job.get_id()] += 1.0

        # Update lost jobs obs
        lost_jobs: np.ndarray = np.zeros(
            1, dtype=np.float64
        )
        lost_jobs[0] = len(self._lost_jobs)

        #New jobs queue
        new_jobs_queue_recipes: np.ndarray = np.full(
            (self._JOBS_QUEUE_LEN,), fill_value=-1, dtype=np.float64
        )
        new_jobs_tray_capacities: np.ndarray = np.full(
            shape=(self._JOBS_QUEUE_LEN,), fill_value=0, dtype=np.float64
        )

        for j_idx, job in enumerate(self._jobs_queue):
            new_jobs_queue_recipes[j_idx] = min_max_norm(job.get_recipes()[0].get_id(), 0, len(self.available_recipes) - 1)
            new_jobs_tray_capacities[j_idx] = job.get_tray_capacity() / self._MACHINE_TRAY_CAPACITY
        #######################################################################################################
        # update incomplete job pending deadline proportional to recipe processing duration times observation #
        #######################################################################################################
        max_duration = min_duration = 0
        # pending_job_remaining_times: np.ndarray = np.zeros(
        #     self._BUFFER_LEN, dtype=np.float64
        # )

        pending_job_next_recipes: np.ndarray = np.full(
            (self._BUFFER_LEN, self._MAX_NEXT_RECIPES), fill_value=-1, dtype=np.float64
        )

        # for j_idx, job in enumerate(self._pending_jobs):
        #     pending_job_recipes[job.get_id()] = job.get_next_pending_recipe().get_id()
        #     pending_job_remaining_times[job.get_id()] = job.get_remaining_process_time()
        #     # update max and min duration times for normalizing [0, 1]
        #     if pending_job_remaining_times[job.get_id()] > max_duration:
        #         max_duration = pending_job_remaining_times[job.get_id()]
        #     elif pending_job_remaining_times[job.get_id()] < min_duration:
        #         min_duration = pending_job_remaining_times[job.get_id()]
        #
        #     for idx, recipe in enumerate(job.get_pending_recipes()):
        #         if idx < self._MAX_NEXT_RECIPES:
        #             pending_job_next_recipes[j_idx, idx] = recipe.get_id()
        #
        # if not sum(pending_job_remaining_times) == 0:
        #     for job in self._pending_jobs:
        #         pending_job_remaining_times[job.get_id()] = (
        #                                                             pending_job_remaining_times[
        #                                                                 job.get_id()] - min_duration
        #                                                     ) / (max_duration - min_duration)

        pending_jobs_steps_to_deadline = [job.get_steps_to_deadline() for job in self._pending_jobs]
        min_deadline = min(pending_jobs_steps_to_deadline)
        max_deadline = max(pending_jobs_steps_to_deadline)
        pending_jobs_steps_to_deadline = np.array(
            [min_max_norm(x, min_deadline, max_deadline) for x in pending_jobs_steps_to_deadline], dtype=np.float64)

        # for idx, val in enumerate(pending_job_remaining_times):
        #     pending_job_remaining_times[idx] = pending_job_remaining_times[idx] * pending_jobs_steps_to_deadline[idx]

        max_duration = min_duration = 0
        pending_job_remaining_times: np.ndarray = np.zeros(
            self._BUFFER_LEN, dtype=np.float64
        )

        pending_job_recipes: np.ndarray = np.zeros(
            self._BUFFER_LEN, dtype=np.float64
        )

        pending_job_tray_capacities: np.ndarray = np.zeros(
            self._BUFFER_LEN, dtype=np.float64
        )
        p_steps_to_deadline_ratio = min_max_norm_list([j.get_process_time_deadline_ratio() for j in self._pending_jobs])

        # uc_buffer_steps_to_deadline_ratio = np.array(
        #     [j.get_process_time_deadline_ratio() for j in self._uncompleted_jobs_buffer])

        for idx, job in enumerate(self._pending_jobs):
            pending_job_recipes[idx] = min_max_norm(job.get_recipes()[0].get_id(), 0, len(self.available_recipes) - 1)
            pending_job_remaining_times[idx] = job.get_remaining_process_time()
            pending_job_tray_capacities[idx] = job.get_tray_capacity() / self._MACHINE_TRAY_CAPACITY
            # update max and min duration times for normalizing [0, 1]
            if pending_job_remaining_times[idx] > max_duration:
                max_duration = pending_job_remaining_times[idx]
            elif pending_job_remaining_times[idx] < min_duration:
                min_duration = pending_job_remaining_times[idx]

        if not sum(pending_job_remaining_times) == 0:
            for j_idx, job in enumerate(self._pending_jobs):
                pending_job_remaining_times[j_idx] = (
                                                                    pending_job_remaining_times[
                                                                        j_idx] - min_duration
                                                            ) / (max_duration - min_duration)

        pending_jobs_steps_to_deadline = [job.get_steps_to_deadline() for job in self._pending_jobs]
        min_deadline = min(pending_jobs_steps_to_deadline)
        max_deadline = max(pending_jobs_steps_to_deadline)
        pending_jobs_steps_to_deadline = np.array(
            [min_max_norm(x, min_deadline, max_deadline) for x in pending_jobs_steps_to_deadline], dtype=np.float64)

        # for idx, val in enumerate(pending_job_remaining_times):
        #     pending_job_remaining_times[idx] = pending_job_remaining_times[idx] * pending_jobs_steps_to_deadline[idx]

        # uc_jobs_obs_list = self.get_uncompleted_job_obs()
        #uc_jobs_buffer_obs_list = self.get_uncompleted_job_buffer_obs()
        #pending_jobs_recipe_count, uc_jobs_recipe_count, uc_jobs_buffer_recipe_count = self.get_recipe_count_obs()
        ###########################################################
        # return current observation state object for step update #
        ###########################################################
        return {
            # self._PENDING_JOBS_STR: is_pending_jobs,
            #self._MACHINES_STR: is_machines_active_jobs.flatten(),
            self._NEW_JOBS_QUEUE_STR: new_jobs_queue_recipes,
            self._NEW_JOBS_TRAY_CAPACITIES_STR: new_jobs_tray_capacities,
            self._P_JOB_RECIPE_STR: pending_job_recipes,
            self._MACHINES_PENDING_CAPACITY_STR: machine_pending_capacity_utilization,
            self._MACHINES_ACTIVE_CAPACITY_STR: machine_active_capacity_utilization,
            self._MACHINES_ACTIVE_RECIPE_STR: machine_active_recipe,
            self._MACHINE_RECIPES: machine_recipes.flatten(),
            #self._P_JOB_REMAINING_TIMES_STR: pending_job_remaining_times,
            self._P_JOB_PROCESS_TIME_TO_DEADLINE_RATIO: p_steps_to_deadline_ratio,
            self._P_JOB_STEPS_TO_DEADLINE: pending_jobs_steps_to_deadline,
            self._P_JOB_TRAY_CAPACITIES: pending_job_tray_capacities
            # self._P_JOB_NEXT_RECIPES: pending_job_next_recipes.flatten(),
            # self._UC_JOB_RECIPE_STR: uc_jobs_obs_list[0],
            # self._UC_JOB_REMAINING_TIMES: uc_jobs_obs_list[1],
            # self._UC_JOB_BUFFER_RECIPES: uc_jobs_buffer_obs_list[0],
            # self._UC_BUFFER_PROCESS_TIME_TO_DEADLINE_RATIO: uc_jobs_buffer_obs_list[2],
            # self._UC_JOB_BUFFER_REMAINING_TIMES: uc_jobs_buffer_obs_list[1],
            # self._UC_JOB_NEXT_RECIPES: uc_jobs_buffer_obs_list[3].flatten(),
            # self._P_JOB_RECIPE_COUNT_STR: pending_jobs_recipe_count,
            # # self._UC_JOB_RECIPE_COUNT_STR: uc_jobs_recipe_count,
            # self._UC_JOB_BUFFER_RECIPE_COUNT: uc_jobs_buffer_recipe_count,
            # self._LOST_JOBS_COUNT: lost_jobs
        }

    def get_machine_scheduled_jobs_matrix(self):
        is_machines_pending_jobs: np.ndarray = np.zeros(
            (len(self._machines), self._BUFFER_LEN), dtype=np.float64
        )
        for machine in self._machines:
            for job in machine.get_pending_jobs():
                is_machines_pending_jobs[machine.get_id(), int(job.get_id())] += 1.0

        return is_machines_pending_jobs

    @staticmethod
    def get_actual_process_time_to_deadline_ratio(jobs_list: list[Job]) -> list[float]:
        steps_to_deadline_ratio = [j.get_process_time_deadline_ratio() for j in jobs_list]
        return steps_to_deadline_ratio

    def get_pending_jobs(self):
        return self._pending_jobs

    def get_uncompleted_jobs(self):
        return self._uncompleted_jobs

    def get_uncompleted_jobs_buffer(self):
        return self._uncompleted_jobs_buffer

    def get_machines(self):
        return self._machines

    def get_buffer_size(self) -> int:
        return self._BUFFER_LEN

    def get_available_recipes_count(self) -> int:
        return len(self.available_recipes)

    def get_max_next_recipes(self) -> int:
        return self._MAX_NEXT_RECIPES

    def get_jobs_completed_on_time(self):
        return self._jobs_completed_per_step_on_time

    def get_jobs_completed_not_on_time(self):
        return self._jobs_completed_per_step_not_on_time

    def get_tardiness_percentage(self):
        # print(f"{self._jobs_completed_per_step_not_on_time} {self._jobs_completed_per_step_on_time}")
        if self._jobs_completed_per_step_not_on_time == 0 and self._jobs_completed_per_step_on_time == 0:
            return 0
        return self._jobs_completed_per_step_not_on_time / (self._jobs_completed_per_step_on_time + self._jobs_completed_per_step_not_on_time) * 100

    def get_jobs_time_past_deadline(self):
        return self._late_jobs_time_past_deadline

    def get_avg_time_past_deadline(self):
        if self._jobs_completed_per_step_not_on_time == 0:
            return 0

        return sum(self._late_jobs_time_past_deadline) / self._jobs_completed_per_step_not_on_time

    def get_uncompleted_job_obs(self) -> list[np.ndarray]:
        max_duration = min_duration = 0
        uc_job_remaining_times: np.ndarray = np.zeros(
            len(self._uncompleted_jobs), dtype=np.float64
        )

        uc_job_recipes: np.ndarray = np.zeros(
            len(self._uncompleted_jobs), dtype=np.float64
        )

        for idx, job in enumerate(self._uncompleted_jobs):
            uc_job_recipes[idx] = job.get_recipes()[0].get_id()
            uc_job_remaining_times[idx] = job.get_remaining_process_time()
            # update max and min duration times for normalizing [0, 1]
            if uc_job_remaining_times[idx] > max_duration:
                max_duration = uc_job_remaining_times[idx]
            elif uc_job_remaining_times[idx] < min_duration:
                min_duration = uc_job_remaining_times[idx]

        if not sum(uc_job_remaining_times) == 0:
            for idx, job in enumerate(self._uncompleted_jobs):
                uc_job_remaining_times[idx] = ((uc_job_remaining_times[idx] - min_duration)
                                               / (max_duration - min_duration))

        if self._uncompleted_jobs:
            uc_jobs_steps_to_deadline = [job.get_steps_to_deadline() for job in self._uncompleted_jobs]
            min_deadline = min(uc_jobs_steps_to_deadline)
            max_deadline = max(uc_jobs_steps_to_deadline)
            uc_jobs_steps_to_deadline = np.array(
                [min_max_norm(x, min_deadline, max_deadline) for x in uc_jobs_steps_to_deadline], dtype=np.float64)

            for idx, val in enumerate(uc_job_remaining_times):
                uc_job_remaining_times[idx] = uc_job_remaining_times[idx] * uc_jobs_steps_to_deadline[idx]

        return [uc_job_recipes, uc_job_remaining_times]

    def get_uncompleted_job_buffer_obs(self) -> list[np.ndarray]:
        max_duration = min_duration = 0
        uc_job_buffer_remaining_times: np.ndarray = np.full(
            self._BUFFER_LEN, fill_value=-1, dtype=np.float64
        )

        uc_job_buffer_recipes: np.ndarray = np.full(
            self._BUFFER_LEN, fill_value=-1, dtype=np.float64
        )

        uc_job_buffer_process_time_deadline_ratio: np.ndarray = np.full(
            self._BUFFER_LEN, fill_value=-1, dtype=np.float64
        )

        uc_job_buffer_next_recipes: np.ndarray = np.full(
            (self._BUFFER_LEN, self._MAX_NEXT_RECIPES), fill_value=-1, dtype=np.float64
        )

        for idx, job in enumerate(self._uncompleted_jobs_buffer):
            uc_job_buffer_recipes[idx] = job.get_pending_recipes()[0].get_id()
            uc_job_buffer_remaining_times[idx] = job.get_remaining_process_time()
            uc_job_buffer_process_time_deadline_ratio[idx] = job.get_process_time_deadline_ratio()
            # update max and min duration times for normalizing [0, 1]
            if uc_job_buffer_remaining_times[idx] > max_duration:
                max_duration = uc_job_buffer_remaining_times[idx]
            elif uc_job_buffer_remaining_times[idx] < min_duration:
                min_duration = uc_job_buffer_remaining_times[idx]

            for r_idx, recipe in enumerate(job.get_pending_recipes()):
                if r_idx < self._MAX_NEXT_RECIPES:
                    uc_job_buffer_next_recipes[idx, r_idx] = recipe.get_id()

        if not sum(uc_job_buffer_remaining_times) == 0:
            for idx, job in enumerate(self._uncompleted_jobs_buffer):
                uc_job_buffer_remaining_times[idx] = ((uc_job_buffer_remaining_times[idx] - min_duration)
                                               / (max_duration - min_duration))

        if self._uncompleted_jobs_buffer:
            uc_jobs_steps_to_deadline = [job.get_steps_to_deadline() for job in self._uncompleted_jobs_buffer]
            min_deadline = min(uc_jobs_steps_to_deadline)
            max_deadline = max(uc_jobs_steps_to_deadline)
            uc_jobs_steps_to_deadline = np.array(
                [min_max_norm(x, min_deadline, max_deadline) for x in uc_jobs_steps_to_deadline], dtype=np.float64)

            for idx, val in enumerate(uc_job_buffer_remaining_times[:len(self._uncompleted_jobs_buffer)]):
                uc_job_buffer_remaining_times[idx] = uc_job_buffer_remaining_times[idx] * uc_jobs_steps_to_deadline[idx]

        return [uc_job_buffer_recipes, uc_job_buffer_remaining_times, uc_job_buffer_process_time_deadline_ratio, uc_job_buffer_next_recipes]

    def get_recipe_count_obs(self) -> list[np.ndarray]:
        uc_jobs_buffer_recipe_count: np.ndarray = np.zeros(
            self._BUFFER_LEN, dtype=np.float64
        )

        uc_jobs_recipe_count: np.ndarray = np.zeros(
            len(self._uncompleted_jobs), dtype=np.float64
        )

        pending_jobs_recipe_count: np.ndarray = np.zeros(
            self._BUFFER_LEN, dtype=np.float64
        )

        for idx, job in enumerate(self._uncompleted_jobs):
            uc_jobs_recipe_count[idx] = len(job.get_pending_recipes())

        for idx, job in enumerate(self._uncompleted_jobs_buffer):
            uc_jobs_buffer_recipe_count[idx] = len(job.get_pending_recipes())

        for idx, job in enumerate(self._pending_jobs):
            pending_jobs_recipe_count[idx] = len(job.get_pending_recipes())

        return [pending_jobs_recipe_count, uc_jobs_recipe_count, uc_jobs_buffer_recipe_count]

    def set_termination_reward(self, reward=-1000):
        self._termination_reward = reward

    def _compute_penalties(self) -> float:
        """
        Calculate reward penalties based on the deadlines
        Helper private method for __compute_reward(), which is a helper private method for step()
        :return: sum of reward penalties
        """
        pending_past_deadline = 0
        inprogress_past_deadline = 0
        # Pending
        for j in self._pending_jobs:
            steps_to_deadline = j.get_steps_to_deadline()
            if steps_to_deadline < 0:
                pending_past_deadline += 1
        # In progress
        for m in self._machines:
            for j in m.get_active_jobs():
                steps_to_deadline = j.get_steps_to_deadline()
                if steps_to_deadline < 0:
                    inprogress_past_deadline += 1

        penalty = (
            self._REWARD_WEIGHTS[self.DEADLINE_EXCEEDED_STR]
            * 0.7
            * inprogress_past_deadline
            + self._REWARD_WEIGHTS[self.DEADLINE_EXCEEDED_STR] * pending_past_deadline * 1.5
        )

        # print(TextColors.RED+"IPPD: "+TextColors.RESET,inprogress_past_deadline)
        # print(TextColors.RED+"PPD: "+TextColors.RESET,pending_past_deadline)
        # print(TextColors.RED+"Penalty: "+TextColors.RESET,penalty)
        return penalty

    def _compute_reward(self) -> float:
        """
        Compute step reward based on minimizing of tardiness and maximizing of machine efficiency, and
        increment episode reward sum for callback graphing of the training performance.
        Helper private method for the overridden env step() method
        :return: total sum of proportional rewards for the step, including all proportional penalties
        """
        # init reward with sum of reward for each completed job, on and not on time, since the previous step
        reward: float = (
            self._REWARD_WEIGHTS[self.JOB_COMPLETED_ON_TIME_STR]
            * self._jobs_completed_per_step_on_time
            + self._REWARD_WEIGHTS[self.JOB_COMPLETED_NOT_ON_TIME_STR]
            * self._jobs_completed_per_step_not_on_time
        )
        # print(TextColors.RED+"COT: "+TextColors.RESET,self._jobs_completed_per_step_on_time)
        # print(TextColors.RED+"CNOT: "+TextColors.RESET,self._jobs_completed_per_step_not_on_time)
        # This is a Delayed Reward, since it comes from the completion of a job that was started in the past

        # This should be reset per episode I think
        # self._jobs_completed_per_step_on_time = (
        #     self._jobs_completed_per_step_not_on_time
        # ) = 0

        return reward + self._compute_penalties() # + self.get_avg_time_past_deadline()

    def _compute_step_reward(self, action) -> float:
        '''
        Checks number of pending jobs past deadline and those not, then gives positive reward if no pending jobs are past deadline.
        @method _compute_pending_job_penalty checks that no-op is used correctly, by making sure it's only used when all pending jobs
        cannot be assigned at the moment
        '''
        reward = 0
        steps_to_deadline = [-1 if job.get_steps_to_deadline() <= 0 else 0 for job in self._pending_jobs]
        #steps_to_deadline = [job.get_steps_to_deadline() for job in self._pending_jobs]
        #reward = self._compute_penalties()
        return (reward + sum(steps_to_deadline) / len(steps_to_deadline)
                + self._compute_pending_job_penalty(action)

                # + self._compute_machine_utilization_reward()
                )

    def _compute_machine_start_reward(self, machine: Machine) -> float:
        reward = 0
        active_recipe = machine.get_active_recipe()
        for job in machine.get_active_jobs():
            if job.get_steps_to_deadline() - active_recipe.get_process_time() >= 0:
                reward += self._REWARD_WEIGHTS[self.JOB_COMPLETED_ON_TIME_STR] * job.get_tray_capacity() / machine.get_tray_capacity()

        if self._can_pending_jobs_be_assigned() and machine.get_active_tray_capacity() >= 5: #40 is the current max job tray size
            reward += self._REWARD_WEIGHTS[self.MACHINE_UNDER_UTILIZED_STR]
        return reward
    def _compute_job_completed_reward(self, job: Job) -> float:
        reward = 0
        p_steps_to_deadline = [j.get_steps_to_deadline() for j in self._pending_jobs]
        reward = (job.get_steps_to_deadline() * 0.02) + (sum(p_steps_to_deadline) * 0.01)

        return reward

    def _compute_pending_job_penalty(self, action) -> float:
        """Function to check if any of pending jobs is assignable, if it is assignable return negative reward"""
        # get job with the least deadline
        p_steps_to_deadline_ratio = np.array([j.get_process_time_deadline_ratio() for j in self._pending_jobs])
        uc_buffer_steps_to_deadline_ratio = np.array([j.get_process_time_deadline_ratio() for j in self._uncompleted_jobs_buffer])
        p_min_job_idx = np.argmax(p_steps_to_deadline_ratio)
        reward = len(self._lost_jobs) * -1

        least_deadline_job = self._pending_jobs[p_min_job_idx]

        if self._uncompleted_jobs_buffer:
            uc_buffer_min_job_idx = np.argmin(uc_buffer_steps_to_deadline_ratio)
            uc_buffer_least_deadline_job = self._uncompleted_jobs_buffer[uc_buffer_min_job_idx]

            if least_deadline_job.get_steps_to_deadline() > uc_buffer_least_deadline_job.get_steps_to_deadline():
                least_deadline_job = uc_buffer_least_deadline_job

        if len(self._machines) * self._BUFFER_LEN < action <= len(self._machines) * self._BUFFER_LEN + len(self._machines):
            # check if started machine could have been assigned a job
            machine_idx = action - (len(self._machines) * self._BUFFER_LEN) - 1
            machine_to_start = self._machines[machine_idx]
            for job in self._pending_jobs:
                if (job.get_next_pending_recipe().get_factory_id() == machine_to_start.get_active_recipe_str()
                        and machine_to_start.get_pending_tray_capacity() >= job.get_tray_capacity()):
                    reward += self._REWARD_WEIGHTS[self.DEADLINE_EXCEEDED_STR]

                    # Also prioritize jobs with multiple recipes
                    if len(job.get_pending_recipes()) > 1:
                        for m_pending_job in machine_to_start.get_pending_jobs():
                            m_p_time_ratio = m_pending_job.get_process_time_deadline_ratio()
                            job_time_ratio = job.get_process_time_deadline_ratio()

                            if job_time_ratio > m_p_time_ratio and machine_to_start.get_pending_tray_capacity() >= job.get_tray_capacity():
                                reward += self._REWARD_WEIGHTS[self.DEADLINE_EXCEEDED_STR] * 1.5
                                break


            # check uncompleted buffer as well
            for job in self._uncompleted_jobs_buffer:
                if (job.get_next_pending_recipe().get_factory_id() == machine_to_start.get_active_recipe_str()
                        and machine_to_start.get_pending_tray_capacity() >= job.get_tray_capacity()):
                    reward += self._REWARD_WEIGHTS[self.DEADLINE_EXCEEDED_STR]

                    # Also prioritize jobs with multiple recipes
                    if len(job.get_pending_recipes()) > 1:
                        for m_pending_job in machine_to_start.get_pending_jobs():
                            m_p_time_ratio = m_pending_job.get_process_time_deadline_ratio()
                            job_time_ratio = job.get_process_time_deadline_ratio()

                            if job_time_ratio > m_p_time_ratio:
                                reward += self._REWARD_WEIGHTS[self.DEADLINE_EXCEEDED_STR] * 1.5
                                break

            return reward

        # print(f"action: {action}")
        if (action != len(self._machines) * self._BUFFER_LEN
                and action <= (len(self._machines) * self._BUFFER_LEN) * 2 + len(self._machines)):
            # if the job is assigned to a specialised machine that can only do that job, reward the agent
            # this makes the agent learn to assign jobs to specialised machines so multi-purpose machines are left to
            # any remaining jobs
            action_selected_machine = None
            if action < len(self._machines) * self._BUFFER_LEN:
                machine_idx = action // self._BUFFER_LEN
                job_idx = action % self._BUFFER_LEN
                action_selected_machine = self._machines[machine_idx]
                action_selected_job = self._pending_jobs[job_idx]

                for idx, job in enumerate(self._pending_jobs):
                    if idx == job_idx:
                        continue

                    if (action_selected_job.get_next_pending_recipe().get_factory_id() == job.get_next_pending_recipe().get_factory_id()
                            and action_selected_job.get_uuid() != job.get_uuid()
                            and action_selected_job.get_process_time_deadline_ratio() < job.get_process_time_deadline_ratio()
                            and job.get_tray_capacity() <= action_selected_machine.get_pending_tray_capacity()
                    ):
                        # same job recipes but picked job with higher deadline
                        reward += -2

                # check for jobs in uncompleted buffer that could have been done
                for idx, job in enumerate(self._uncompleted_jobs_buffer):
                    if (action_selected_job.get_next_pending_recipe().get_factory_id() == job.get_next_pending_recipe().get_factory_id()
                            and action_selected_job.get_factory_id() != job.get_factory_id()
                            and action_selected_job.get_steps_to_deadline() > job.get_steps_to_deadline()
                    ):
                        # same job recipes but picked job with higher deadline and in pending buffer
                        reward += self._REWARD_WEIGHTS[self.DEADLINE_EXCEEDED_STR]
            else:
                action_offset = len(self._machines) * self._BUFFER_LEN + len(self._machines) + 1
                decoded_action = action - action_offset
                machine_idx = decoded_action // self._BUFFER_LEN
                job_idx = decoded_action % self._BUFFER_LEN
                action_selected_machine = self._machines[machine_idx]
                if job_idx >= len(self._uncompleted_jobs_buffer):
                    reward += self._REWARD_WEIGHTS[self.ILLEGAL_ACTION_STR]
                    return reward

                action_selected_job = self._uncompleted_jobs_buffer[job_idx]

                for idx, job in enumerate(self._uncompleted_jobs_buffer):
                    if idx == job_idx:
                        continue

                    if (action_selected_job.get_next_pending_recipe().get_factory_id() == job.get_next_pending_recipe().get_factory_id()
                            and action_selected_job.get_factory_id() != job.get_factory_id()
                            and action_selected_job.get_process_time_deadline_ratio() < job.get_process_time_deadline_ratio()
                    ):
                        # same job recipes but picked job with higher deadline
                        reward += -2

            if len(action_selected_machine.get_valid_recipes()) == 1:
                reward += 0
            else:
                # check if there is a specialised machine available
                found_specialised_machine = False
                min_recipes = math.inf
                for idx, machine in enumerate(self._machines):
                    if idx == machine_idx:
                        continue

                    if (
                            len(machine.get_valid_recipes()) < len(action_selected_machine.get_valid_recipes())
                            and machine.is_available()
                            and machine.can_perform_job(action_selected_job)
                            and machine.get_pending_tray_capacity() >= action_selected_job.get_tray_capacity()
                    ):
                        #print(f"Another specialised machine could have done job: {action_selected_job}")
                        reward += -2
                        found_specialised_machine = True
                # no specialised machine, agent should be rewarded for taking this action
                if not found_specialised_machine:
                    reward += 0

            # Also if all jobs are of the same type, the agent should be punished for picking a job with a higher deadline
            # than an already existing job

            # if the deadline will pass after executing this job, give a negative reward
            # if action_selected_job.get_steps_to_deadline() - action_selected_job.get_recipes()[0].get_process_time() < 0:
            #     reward += self._REWARD_WEIGHTS[self.DEADLINE_EXCEEDED_STR]

            return reward

        #if action is no-op, check if any available machine could have been started
        if action == len(self._machines) * self._BUFFER_LEN:
            for machine in self._machines:
                if machine.is_available():
                    jobs_assignable = False
                    for job in self._pending_jobs:
                        if machine.can_perform_job(job):
                            jobs_assignable = True
                            reward += self._REWARD_WEIGHTS[self.DEADLINE_EXCEEDED_STR]
                    for job in self._uncompleted_jobs_buffer:
                        if machine.can_perform_job(job):
                            jobs_assignable = True
                            reward += self._REWARD_WEIGHTS[self.DEADLINE_EXCEEDED_STR]

                    if not jobs_assignable and len(machine.get_pending_jobs()) > 0:
                        # machine should have been started
                        reward += self._REWARD_WEIGHTS[self.DEADLINE_EXCEEDED_STR]

        for machine in self._machines:
            if machine.is_available() and machine.can_perform_job(least_deadline_job):
                reward += self._REWARD_WEIGHTS[self.DEADLINE_EXCEEDED_STR]

        return reward

    def _compute_machine_utilization_reward(self):
        """
        Function to check if there are pending jobs and machines idle
        """
        for machine in self._machines:
            if machine.is_available():
                # check pending jobs if they are assignable to this machine
                # if assignable negative reward, else 0
                for job in self._pending_jobs:
                    if machine.can_perform_job(job):
                        print('Machine utilization punishment')
                        return self._REWARD_WEIGHTS[self.MACHINE_IDLE_STR]

        return 0

    def _can_pending_jobs_be_assigned(self) -> bool:
        for machine in self._machines:
            for job in self._pending_jobs:
                if machine.is_available() and machine.can_perform_job(job):
                    if (machine.get_active_recipe() is None or
                            machine.get_active_recipe().get_factory_id() == job.get_next_pending_recipe().get_factory_id()):
                        return True

        return False

    def _update_deadlines(self, time_delta: float):
        for j in self._pending_jobs:
            #j.update_steps_to_deadline(difference=time_delta * -1)
            j.sync_steps_to_deadline(self.factory_time)
        for j in self._uncompleted_jobs_buffer:
            #j.update_steps_to_deadline(difference=time_delta * -1)
            j.sync_steps_to_deadline(self.factory_time)
        for j in self._lost_jobs:
            #j.update_steps_to_deadline(difference=time_delta * -1)
            j.sync_steps_to_deadline(self.factory_time)
        for m in self._machines:
            for j in m.get_active_jobs():
                j.update_steps_to_deadline(difference=time_delta * -1)

    def _is_jobs_done(self) -> bool:
        return len(self._completed_jobs) == self._BUFFER_LEN

    def _update_factory_env_state(self, no_op_time: int) -> float:
        """
        Check and update the status of machines and jobs being processed since the last step.
        Updates time in total factory process, and each Machine object's total activity or idleness.
        Helper private method for the overridden env step() method
        :return: conditional for if all jobs in the buffer are completed
        1. Get machines-jobs
        2. If available machine -> +1 timestep and return
        3. If machines occupied ->
            3.1 Find smallest value to free up a machine (soonest finishing job)
            3.2 Advance "clock" in timesteps, and also the jobs will progress that time
        """
        na = 0
        min_time = math.inf  # NOTE: Set to infinity (is it with numpy?)
        time_delta = 0  # this is the amount of time i will move towards the future

        available = [1 for m in self._machines if m.is_available()]
        na = sum(available)
        job_completed_reward = 0

        if na == 0:
            # I search for the min completion time - This could probably be done with list comprehension
            for machine in self._machines:
                if not machine.is_available() and len(machine.get_active_jobs()) > 0:
                    # print("compare: ", machine.get_active_jobs()[0].get_steps_to_recipe_complete() )
                    # print("with:",min_time)
                    if (
                        machine.get_active_jobs()[0].get_steps_to_recipe_complete()
                        < min_time
                    ):
                        min_time = machine.get_active_jobs()[
                            0
                        ].get_steps_to_recipe_complete()
                        # Even if there are more than 1 jobs, they will all start and finish at the same time
            # print("Min time: ", min_time)
            time_delta = min_time
        else:
            if no_op_time > 0:
                time_delta = no_op_time
            else:
                time_delta = 1

        # Then I advance all the jobs in progress and the environment
        self.factory_time += time_delta
        #print(f'Factory time {self.factory_time}')
        for machine in self._machines:
            if not machine.is_available():
                for idx, j in enumerate(machine.get_active_jobs()):
                    j.update_steps_to_recipe_complete(time_delta * -1)
                    j.update_steps_to_deadline(time_delta * -1)
                    if j.get_steps_to_recipe_complete() <= 0:
                        j.set_recipe_completed(j.get_recipe_in_progress())

                        #if job has no recipes left, it is completed
                        if not j.get_pending_recipes():
                            self._completed_jobs.append(
                                self._jobs_in_progress.pop(
                                    self._jobs_in_progress.index((machine, j))
                                )[1]
                            )
                            if j.get_steps_to_deadline() >= 0:
                                self._jobs_completed_per_step_on_time += 1
                                job_completed_reward += self._REWARD_WEIGHTS[self.JOB_COMPLETED_ON_TIME_STR]
                            else:
                                self._jobs_completed_per_step_not_on_time += 1
                                self._late_jobs_time_past_deadline.append(j.get_steps_to_deadline())

                            #machine.remove_job_assignment(job=j)
                            # return an immediate reward for job completion that takes into considering the amount of time to deadline
                            #job_completed_reward = self._compute_job_completed_reward(j)
                        else:
                            # set back the time to deadline for job, since it will be updated in the update_deadlines function
                            j.update_steps_to_deadline(time_delta * 1)
                            # job recipe completed, but job has pending recipes
                            if len(self._uncompleted_jobs_buffer) < self._BUFFER_LEN:
                                self._uncompleted_jobs_buffer.append(j)
                            else:
                                self._lost_jobs.append(j)

                        if idx == len(machine.get_active_jobs()) - 1:
                            machine.remove_completed_jobs()
        # Finally, Update all deadlines based on the time passed
        self._update_deadlines(time_delta)
        return job_completed_reward

    def _init_machine_job(self, selected_machine: Machine, selected_job: Job, from_pending_buffer: bool = True) -> bool:
        """
        Add one pending job to one available machine given one job recipe is valid for given machine.
        Helper private method for Env step() method
        :param selected_machine: the available machine pending job assignment
        :param selected_job: the available job pending machine assignment
        :return: True if machine is assigned new pending job, otherwise False
        """
        if selected_machine.schedule_job(job_to_schedule=selected_job):
            self._jobs_in_progress.append((selected_machine, selected_job))
            if from_pending_buffer:
                self._pending_jobs.remove(selected_job)
            else:
                self._uncompleted_jobs_buffer.remove(selected_job)
            return True
            # print(f'Assigned job {selected_job} to machine {selected_machine.get_factory_id()}')
        return False

    def _check_termination(self):
        if self._time_step >= self._max_steps:
            return True
        if not self.is_evaluation and self.episode_reward_sum < self._termination_reward:
            return True
        return False

    def _calc_noop_time(self):
        times = [
            m.get_active_jobs()[0].get_steps_to_recipe_complete()
            for m in self._machines
            if not m.is_available()
        ]
        # if there's a free machine, check if any jobs are assignable to it
        eligible_job_exists = False
        for m in self._machines:
            if m.is_available():
                eligible_job_exists = (m.can_perform_any_pending_job(self._pending_jobs)
                                       or m.can_perform_any_pending_job(self._uncompleted_jobs_buffer))


        if times:
            if eligible_job_exists:
                return 1
            else:
                return min(times)
        return 1

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray[any]], float, bool, bool, dict[str, str]]:
        """
        Take a single step in the factory environment.
        :param action: the agent's action to take in the step
        :return: (observation, reward, terminated, truncated, info)
        """
        self._time_step += 1
        step_reward = self._compute_step_reward(action)
        # print(f"step_reward: {step_reward}")
        no_op_time = 0
        # print(TextColors.MAGENTA+"Factory Time before step: "+TextColors.RESET,self.factory_time)
        #########################
        #     1-TAKE ACTION     #
        #########################
        if action == len(self._machines) * self._BUFFER_LEN:
            # If No-Op is the selected action - Then we have to jump a # of timesteps ahead in a logic
            # proportion to the jobs in progress
            step_reward += 0
            no_op_time = self._calc_noop_time()
            # NOTE: Have to determine here to send noop time to skip
        # action is a start machine action
        elif len(self._machines) * self._BUFFER_LEN < action <= len(self._machines) * self._BUFFER_LEN + len(self._machines):
            # If Start machines action is selected, we have to start running the pending jobs on all machines
            machine_idx = action - (len(self._machines) * self._BUFFER_LEN) - 1
            machine_to_start = self._machines[machine_idx]

            machine_started = machine_to_start.start()

            if machine_started:
                # reward is the number of the jobs scheduled
                step_reward += self._compute_machine_start_reward(machine_to_start)
            else:
                step_reward += self._REWARD_WEIGHTS[self.MACHINE_UNAVAILABLE_STR]

        elif action < len(self._machines) * self._BUFFER_LEN:
            # If action is for assigning jobs from pending array
            action_selected_machine = self._machines[
                action // self._BUFFER_LEN
            ]  # get action selected machine
            # Check Machine Availability
            if action_selected_machine.is_available():
                action_selected_job = self._pending_jobs[
                    action % self._BUFFER_LEN
                ]  # get action selected job
                if self._init_machine_job(
                    selected_machine=action_selected_machine,
                    selected_job=action_selected_job,
                ):
                    step_reward += 1  # NOTE:Check if giving a reward for correct assignment makes sense.
                else:
                    # action selected machine is available but action selected job is invalid for selected machine
                    #print('Invalid job recipe')
                    step_reward += self._REWARD_WEIGHTS[self.INVALID_JOB_RECIPE_STR]
            else:
                # action selected machine is available but action selected job is invalid for selected machine
                #print('Machine unavailable')
                step_reward += self._REWARD_WEIGHTS[self.MACHINE_UNAVAILABLE_STR]
        elif len(self._machines) * self._BUFFER_LEN + len(self._machines) < action <= (len(self._machines) * self._BUFFER_LEN) * 2 + len(self._machines):
            #action is for assigning jobs from uncompleted_buffer
            action_offset = len(self._machines) * self._BUFFER_LEN + len(self._machines) + 1
            decoded_action = action - action_offset
            action_selected_machine = self._machines[
                decoded_action // self._BUFFER_LEN
                ]
            if action_selected_machine.is_available():
                if decoded_action % self._BUFFER_LEN >= len(self._uncompleted_jobs_buffer):
                    step_reward += self._REWARD_WEIGHTS[self.ILLEGAL_ACTION_STR]
                else:
                    action_selected_job = self._uncompleted_jobs_buffer[
                        decoded_action % self._BUFFER_LEN
                        ]
                    if self._init_machine_job(
                            selected_machine=action_selected_machine,
                            selected_job=action_selected_job,
                            from_pending_buffer=False
                    ):
                        step_reward += 1
                    else:
                        step_reward += self._REWARD_WEIGHTS[self.MACHINE_UNAVAILABLE_STR]
            else:
                step_reward += self._REWARD_WEIGHTS[self.MACHINE_UNAVAILABLE_STR]
        # TODO: remove the `else` block
        # else:
        #     # Action is for moving jobs from uncompleted_jobs list to uncompleted_jobs_buffer
        #     action_offset = (len(self._machines) * self._BUFFER_LEN) * 2 + len(self._machines) + 1
        #     decoded_action = action - action_offset
        #     if len(self._uncompleted_jobs) < decoded_action + 1:
        #         step_reward += self._REWARD_WEIGHTS[self.ILLEGAL_ACTION_STR]
        #     else:
        #         # check if _uncompleted_jobs_buffer is not full, otherwise negative reward
        #         if len(self._uncompleted_jobs_buffer) < self._BUFFER_LEN:
        #             #move job to buffer
        #             job_to_move = self._uncompleted_jobs.pop(decoded_action)
        #             self._uncompleted_jobs_buffer.append(job_to_move)
        #             step_reward += 1.0 # TODO: find better reward depending on if the right job was moved
        #         else:
        #             step_reward += self._REWARD_WEIGHTS[self.ILLEGAL_ACTION_STR]
        #########################
        #     2-UPDATE ENV      #
        #########################
        self._update_factory_env_state(no_op_time=no_op_time)
        self.callback_step_tardiness = self.get_tardiness_percentage()
        #########################
        #     3-CALC REWARD     #
        #########################
        #state_reward = self._compute_reward()
        reward = step_reward
        # print(TextColors.RED+"Reward:"+TextColors.RESET,reward)
        self.episode_reward_sum += reward
        self.callback_step_reward = reward
        #########################
        #  4-CHECK TERMINATION  #
        #########################
        is_terminated = self._check_termination()
        # if is_terminated: #No terminal rewards any longer
        #     state_reward = self._compute_reward()
        #     reward += state_reward
        #########################
        #    5-UPDATE BUFFERS    #
        #########################
        self.update_buffer()
        self.update_uncompleted_buffer()
        # print(TextColors.MAGENTA+"Factory Time after step: "+TextColors.RESET,self.factory_time)
        #########################
        #        6-RETURN       #
        #########################
        return (
            self.get_capacity_obs(),
            reward,
            is_terminated,
            False,  # NOTE: Check truncation conditions
            {"INFO": str(reward) + "," + str(self.episode_reward_sum),
             "JOBS_COMPLETED_ON_TIME": self._jobs_completed_per_step_on_time,
             "JOBS_NOT_COMPLETED_ON_TIME": self._jobs_completed_per_step_not_on_time,
             "AVG_TARDINESS_OF_LATE_JOBS": self.get_avg_time_past_deadline(),
             "CURRENT_TIME": self.factory_time,
             "UNCOMPLETED_JOBS_BUFFER": len(self._uncompleted_jobs_buffer),
             "LOST_JOBS": len(self._lost_jobs)
             },
        )

    def update_buffer(self):
        """
        This method will create the new jobs to add to the buffer so that it is 'never ending'
        """
        # print(TextColors.RED+"\n\n###################################"+TextColors.RESET)
        # print(TextColors.RED+"#       Updating the buffer      #"+TextColors.RESET)
        # print(TextColors.RED+"###################################\n"+TextColors.RESET)
        if len(self._pending_jobs) < self._BUFFER_LEN:
            # dif = self._BUFFER_LEN - len(self._pending_jobs)
            # print("\n\nupdate_buffer - Buffer should be updated with ",dif," new jobs")
            # print("Length Pending Jobs: ",len(self._pending_jobs))
            # print("First job: ")
            # print(self._pending_jobs[0])
            r = np.random.choice(np.array(self.available_recipes), p=self.available_recipe_probs)
            r2 = random.choice(self.available_recipes, )
            use_multiple_recipes = False if self.MAX_RECIPES_IN_ENV_SYSTEM == 1 else np.random.randint(low=0,
                                                                                                       high=10) % 2 == 0

            # NOTE:Note that “process time” should be 20%-28% of the difference between deadline-arrival
            job_to_move = self._jobs_queue.pop(0)
            new_job = create_job(
                recipes=[r] if not use_multiple_recipes else [r, r2],
                factory_id="J" + str(self.factory_time % self._BUFFER_LEN),
                process_id=self.factory_time % self._BUFFER_LEN,
                deadline=0,
                factory_time=self.factory_time,
            )
            # print(TextColors.YELLOW+"Buffer updated with job: "+TextColors.RESET)
            # print(new_job)
            if self._fresh_arrival_time:
                job_to_move.update_creation_time(self.factory_time) # This makes it seem like the job was created the time it was inserted into the buffer
            if len(self._jobs_queue) < self._JOBS_QUEUE_LEN:
                self._jobs_queue.append(new_job)

            self._pending_jobs.append(job_to_move)

    def update_uncompleted_buffer(self):
        """
        This method will move any jobs in lost_jobs to the uncompleted buffer.
        TODO: Update so only jobs that are within deadline are movable
        """
        if len(self._lost_jobs) > 0 and len(self._uncompleted_jobs_buffer) < self._BUFFER_LEN:
            job = self._lost_jobs.pop(0)
            self._uncompleted_jobs_buffer.append(job)


    def render(self):
        """
        Print the state of the environment at current step
        """
        active_jobs: np.ndarray = np.zeros(
            (len(self._machines), self._BUFFER_LEN), dtype=np.float64
        )
        for machine in self._machines:
            for job in machine.get_active_jobs():
                active_jobs[machine.get_id(), job.get_id()] = 1.0

        print(TextColors.YELLOW + "Machine occupancy:" + TextColors.RESET)
        print(TextColors.GREEN + "       J0  J1  J2" + TextColors.RESET)
        for i in range(len(self._machines)):
            print(TextColors.GREEN + "M", i, " " + TextColors.RESET + "[ ", end="")
            for j in range(len(active_jobs[i])):
                if active_jobs[i][j] == 1:
                    # print(TextColors.GREEN+"M",i," "+TextColors.RESET,machines_matrix[i])
                    print(TextColors.CYAN + "1.  " + TextColors.RESET, end="")
                else:
                    print("0.  ", end="")
            print("]")
        print()

    def reset(
        self, seed: int = None, options: str = None
    ) -> tuple[dict[str, np.ndarray[any]], dict[str, str]]:
        # print cum rewards
        # print factory time
        # print(f'factory time: {self.factory_time}')
        # print(TextColors.RED+'Pre-Reset: Episode reward sum:'+TextColors.RESET,self.episode_reward_sum)
        """
        Reset the environment state
        """
        self._time_step = 0
        self.factory_time = 0.0

        jobs: list[Job] = [
            create_job(
                recipes=[np.random.choice(np.array(self.available_recipes), p=self.available_recipe_probs)],
                factory_id=f"J{i}",
                process_id=i % self._BUFFER_LEN,
                deadline=0,
                factory_time=0
            ) for i in range(self._JOBS_QUEUE_LEN)
        ]

        self.episode_reward_sum = 0  # for callback graphing train performance
        self.callback_flag_termination = (
            False  # for callback graphing train performance
        )
        self.callback_step_reward = 0  # for callback graphing train performance

        for machine in self._machines:
            machine.reset()

        # self._machines: list[Machine] = self._total_machines_available.copy()[
        #     : self._MAX_MACHINES
        # ]

        for job in self._jobs_queue:
            job.reset()
        self._pending_jobs = jobs.copy()[: self._BUFFER_LEN]
        self._jobs_queue = jobs.copy()[self._BUFFER_LEN: ]
        self._jobs_completed_per_step_on_time = 0
        self._jobs_completed_per_step_not_on_time = 0
        self._jobs_in_progress = []
        self._completed_jobs = []
        self._late_jobs_time_past_deadline = []
        self._uncompleted_jobs_buffer = []
        self._lost_jobs = []


        return self.get_capacity_obs(), {}

    def close(self) -> None:
        """
        Close the environment
        """
        self.reset()
