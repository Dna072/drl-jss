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
from custom_environment.recipe_factory import create_recipe
from custom_environment.job_factory import create_job
from custom_environment.utils import min_max_norm

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
    NEUTRAL_STR: str = "NEUTRAL"

    MAX_RECIPES_IN_ENV_SYSTEM: int = 2
    r1 = create_recipe(
        factory_id="R1_ID", process_time=30.0, process_id=0, recipe_type="R1"
    )
    r2 = create_recipe(
        factory_id="R2_ID", process_time=300.0, process_id=1, recipe_type="R2"
    )
    available_recipes = [r1, r2]

    #####################
    # private constants #
    #####################

    _BUFFER_LEN: int = 3
    _NO_OP_SPACE: int = 1
    _MAX_MACHINES: int = 2
    _RECIPES_LENGTH: dict[str, int] = {"R1": 1, "R2": 10}
    _REWARD_WEIGHTS: dict[str, int] = {
        NO_OP_STR: 0,
        MACHINE_IDLE_STR: -1,
        MACHINE_UNAVAILABLE_STR: -5,
        JOB_COMPLETED_ON_TIME_STR: 10,
        JOB_COMPLETED_NOT_ON_TIME_STR: 0,
        JOB_ASSIGNED_STR: 1,
        INVALID_JOB_RECIPE_STR: -5,
        DEADLINE_EXCEEDED_STR: -5,
        ILLEGAL_ACTION_STR: -5,
        NEUTRAL_STR: 0,
    }

    _REWARD_TIME_PENALTIES: dict[str, dict[str, int | float]] = {
        "10_hrs": {"in_s": 36_000, "weight": 0.4},
        "24_hrs": {"in_s": 86_400, "weight": 0.8},
    }

    # observation space constant dict keys
    _PENDING_JOBS_STR: str = "pending_jobs"
    _RECIPE_TYPES_STR: str = "recipes"
    _MACHINES_STR: str = "machines"
    _P_JOB_REMAINING_TIMES_STR: str = "pending_job_remaining_times"
    _P_JOB_STEPS_TO_DEADLINE: str = "pending_job_steps_to_deadline"
    _IP_JOB_REMAINING_TIMES_STR: str = "inprogress_job_remaining_times"

    _METADATA: dict[int, str] = {0: "vector", 1: "human"}

    def __init__(
        self, machines: list[Machine], jobs: list[Job], max_steps: int = 10_000, is_evaluation: bool = False
    ) -> None:
        """
        FactoryEnv class constructor method using gym.Space objects for action and observation space
        :param machines: array of Machine instances
        :param jobs: array of Job instances
        :param max_steps: maximum episode length
        """
        super(FactoryEnv, self).__init__()
        self._total_machines_available: list[Machine] = machines
        self._machines: list[Machine] = self._total_machines_available.copy()[
            : self._MAX_MACHINES
        ]  # restricted len for max machines being used
        self.is_evaluation: bool = is_evaluation
        self._jobs: list[Job] = jobs
        self._pending_jobs: list[Job] = self._jobs.copy()[
            : self._BUFFER_LEN
        ]  # buffer of restricted len
        self._jobs_in_progress: list[tuple[Machine, Job]] = []
        self._completed_jobs: list[Job] = []
        self._jobs_completed_per_step_on_time: int = 0
        self._jobs_completed_per_step_not_on_time: int = 0

        # variable to keep track of time past deadline for jobs completed after their deadline
        self._late_jobs_time_past_deadline: list[float] = []

        self._max_steps: int = max_steps
        self._time_step: int = 0
        self._step_datetime: str | None = None

        self.factory_time: float = (
            0.0  # NOTE: This is the variable that tracks processing time
        )
        self._termination_reward: float = -1000  # NOTE: Check if applicable
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
            len(self._machines) * self._BUFFER_LEN + self._NO_OP_SPACE
        )

        #####################
        # observation space #
        #####################

        pending_jobs_space: gym.spaces.Box = gym.spaces.Box(
            low=0, high=1, shape=(self._BUFFER_LEN,), dtype=np.float64
        )  # binary vector for representing if a job is pending for job assignment

        machine_space: gym.spaces.Box = gym.spaces.Box(
            low=0,
            high=1,
            shape=(len(self._machines) * self._BUFFER_LEN,),
            dtype=np.float64,
        )  # binary matrix for mapping machines to jobs they are processing
        pending_job_remaining_times_space: gym.spaces.Box = gym.spaces.Box(
            low=0, high=1, shape=(self._BUFFER_LEN,), dtype=np.float64
        )  # normalized vector for jobs pending deadline proportional to recipe processing duration times

        pending_job_steps_to_deadline_space: gym.spaces.Box = gym.spaces.Box(
            low=0, high=1, shape=(self._BUFFER_LEN, ), dtype=np.float64
        )  # normalized vector (using Min-max scaling [0,1]) for steps to deadline for jobs in buffer

        self.observation_space: gym.spaces.Dict = gym.spaces.Dict(
            {
                # self._PENDING_JOBS_STR: pending_jobs_space,
                self._MACHINES_STR: machine_space,
                self._P_JOB_REMAINING_TIMES_STR: pending_job_remaining_times_space,
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
            (len(self._machines), self._BUFFER_LEN), dtype=np.float64
        )
        for machine in self._machines:
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

    def get_pending_jobs(self):
        return self._pending_jobs

    def get_machines(self):
        return self._machines

    def get_buffer_size(self):
        return self._BUFFER_LEN

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

    def _compute_job_completed_reward(self, job: Job) -> float:
        reward = 0
        p_steps_to_deadline = [j.get_steps_to_deadline() for j in self._pending_jobs]
        reward = (job.get_steps_to_deadline() * 0.02) + (sum(p_steps_to_deadline) * 0.01)

        return reward

    def _compute_pending_job_penalty(self, action) -> float:
        """Function to check if any of pending jobs is assignable, if it is assignable return negative reward"""
        # get job with the least deadline
        p_steps_to_deadline = np.array([j.get_steps_to_deadline() for j in self._pending_jobs])
        min_job_idx = np.argmin(p_steps_to_deadline)

        least_deadline_job = self._pending_jobs[min_job_idx]

        # print(f"action: {action}")
        if action != len(self._machines) * self._BUFFER_LEN:
            # if the job is assigned to a specialised machine that can only do that job, reward the agent
            # this makes the agent learn to assign jobs to specialised machines so multi-purpose machines are left
            # any remaining jobs
            machine_idx = action // self._BUFFER_LEN
            job_idx = action % self._BUFFER_LEN
            action_selected_machine = self._machines[machine_idx]
            action_selected_job = self._pending_jobs[job_idx]

            for idx, job in enumerate(self._pending_jobs):
                if idx == job_idx:
                    continue

                if (action_selected_job.get_recipes() == job.get_recipes()
                        and action_selected_job.get_factory_id() != job.get_factory_id()
                        and action_selected_job.get_steps_to_deadline() > job.get_steps_to_deadline()
                ):
                    # same job recipes
                    return -2

            if len(action_selected_machine.get_valid_recipes()) == 1:
                return 1
            else:
                # check if there is a specialised machine available
                for idx, machine in enumerate(self._machines):
                    if idx == machine_idx:
                        continue

                    if (
                            len(machine.get_valid_recipes()) == 1
                            and machine.is_available()
                            and machine.can_perform_job(action_selected_job)
                    ):
                        #print(f"Another specialised machine could have done job: {action_selected_job}")
                        return -1

            # Also if all jobs are of the same type, the agent should be punished for picking a job with a higher deadline
            # than an already existing job

            return 0

        for machine in self._machines:
            if machine.is_available() and machine.can_perform_job(least_deadline_job):
                return self._REWARD_WEIGHTS[self.DEADLINE_EXCEEDED_STR]

        return 0

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
                        return self._REWARD_WEIGHTS[self.MACHINE_IDLE_STR]

        return 0

    def _update_deadlines(self, time_delta: float):
        for j in self._pending_jobs:
            j.update_steps_to_deadline(difference=time_delta * -1)
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
                if not machine.is_available():
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
        for machine in self._machines:
            if not machine.is_available():
                for j in machine.get_active_jobs():
                    j.update_steps_to_recipe_complete(time_delta * -1)
                    if j.get_steps_to_recipe_complete() <= 0:
                        j.set_recipe_completed(j.get_recipes_in_progress()[0])
                        self._completed_jobs.append(
                            self._jobs_in_progress.pop(
                                self._jobs_in_progress.index((machine, j))
                            )[1]
                        )
                        if j.get_steps_to_deadline() >= 0:
                            self._jobs_completed_per_step_on_time += 1
                        else:
                            self._jobs_completed_per_step_not_on_time += 1
                            self._late_jobs_time_past_deadline.append(j.get_steps_to_deadline())

                        machine.remove_job_assignment(job=j)
                        # return an immediate reward for job completion that takes into considering the amount of time to deadline
                        job_completed_reward = self._compute_job_completed_reward(j)
        # Finally, Update all deadlines based on the time passed
        self._update_deadlines(time_delta)
        return job_completed_reward

    def _init_machine_job(self, selected_machine: Machine, selected_job: Job) -> bool:
        """
        Add one pending job to one available machine given one job recipe is valid for given machine.
        Helper private method for Env step() method
        :param selected_machine: the available machine pending job assignment
        :param selected_job: the available job pending machine assignment
        :return: True if machine is assigned new pending job, otherwise False
        """
        if selected_machine.assign_job(job_to_assign=selected_job):
            self._jobs_in_progress.append((selected_machine, selected_job))
            self._pending_jobs.remove(selected_job)
            # print(f'Assigned job {selected_job} to machine {selected_machine.get_factory_id()}')
        return not selected_machine.is_available()

    def _check_termination(self):
        if self.factory_time >= self._max_steps:
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
        if times != []:
            if min(times) // 4 <= 1:
                return 1
            else:
                return min(times) // 4
        return 1

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray[any]], float, bool, bool, dict[str, str]]:
        """
        Take a single step in the factory environment.
        :param action: the agent's action to take in the step
        :return: (observation, reward, terminated, truncated, info)
        """
        is_terminated: bool = False
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
        else:
            # If action wasnt No-Op, then we compute machine-job
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
                    step_reward += self._REWARD_WEIGHTS[self.INVALID_JOB_RECIPE_STR]
            else:
                # action selected machine is available but action selected job is invalid for selected machine
                step_reward += self._REWARD_WEIGHTS[self.MACHINE_UNAVAILABLE_STR]
        #########################
        #     2-UPDATE ENV      #
        #########################
        self._update_factory_env_state(no_op_time=no_op_time)
        self.callback_step_tardiness = self.get_tardiness_percentage()
        #########################
        #     3-CALC REWARD     #
        #########################
        #state_reward = self._compute_reward()
        reward = step_reward  # + job_completed_reward #+ state_reward
        # print(TextColors.RED+"Reward:"+TextColors.RESET,reward)
        self.episode_reward_sum += reward
        self.callback_step_reward = reward
        #########################
        #  4-CHECK TERMINATION  #
        #########################
        is_terminated = self._check_termination()
        if is_terminated:
            state_reward = self._compute_reward()
            reward += state_reward
        #########################
        #    5-UPDATE BUFFER    #
        #########################
        self.update_buffer()
        # print(TextColors.MAGENTA+"Factory Time after step: "+TextColors.RESET,self.factory_time)
        #########################
        #        6-RETURN       #
        #########################
        return (
            self.get_obs(),
            reward,
            is_terminated,
            False,  # NOTE: Check truncation conditions
            {"INFO": str(reward) + "," + str(self.episode_reward_sum),
             "JOBS_COMPLETED_ON_TIME": self._jobs_completed_per_step_on_time,
             "JOBS_NOT_COMPLETED_ON_TIME": self._jobs_completed_per_step_not_on_time,
             "AVG_TARDINESS_OF_LATE_JOBS": self.get_avg_time_past_deadline(),
             "CURRENT_TIME": self.factory_time
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
            pending_idx = [j.get_id() for j in self._pending_jobs]
            for i in range(self._BUFFER_LEN):
                if i not in pending_idx:
                    # Create job
                    r = random.choice(self.available_recipes,)
                    # NOTE:Note that “process time” should be 5%-30% of the difference between deadline-arrival
                    new_job = create_job(
                        recipes=[r],
                        factory_id="J" + str(i),
                        process_id=i,
                        deadline=100 if r.get_recipe_type() == "R1" else 1000,
                        factory_time=self.factory_time,
                    )
                    # print(TextColors.YELLOW+"Buffer updated with job: "+TextColors.RESET)
                    # print(new_job)
                    self._pending_jobs.insert(i, new_job)

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

        for job in self._jobs:
            job.reset()
        self._pending_jobs = self._jobs.copy()[: self._BUFFER_LEN]
        self._jobs_completed_per_step_on_time = 0
        self._jobs_completed_per_step_not_on_time = 0
        self._jobs_in_progress = []
        self._completed_jobs = []
        self._late_jobs_time_past_deadline = []

        return self.get_obs(), {}

    def close(self) -> None:
        """
        Close the environment
        """
        self.reset()
