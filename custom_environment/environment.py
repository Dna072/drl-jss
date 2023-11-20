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

from custom_environment.machine import Machine
from custom_environment.job import Job

from IPython.display import clear_output
from datetime import datetime, timedelta
import plotly.figure_factory as ff
import gymnasium as gym
from time import sleep
import numpy as np


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
    JOB_COMPLETED_ON_TIME_STR: str = "JOB_COMPLETED_ON_TIME"
    JOB_COMPLETED_NOT_ON_TIME_STR: str = "JOB_COMPLETED_NOT_ON_TIME"
    INVALID_JOB_RECIPE_STR: str = "INVALID JOB RECIPE"
    DEADLINE_EXCEEDED_STR: str = "DEADLINE_EXCEEDED"
    ILLEGAL_ACTION_STR: str = "ILLEGAL ACTION"
    NEUTRAL_STR: str = "NEUTRAL"

    MAX_RECIPES_IN_ENV_SYSTEM: int = 2

    #####################
    # private constants #
    #####################

    __BUFFER_LEN: int = 3
    __NO_OP_SPACE: int = 1
    __NUM_GOALS = 2  # minimize tardiness, maximize efficiency

    __MAX_MACHINES: int = 2
    __MAX_MACHINES_ASSIGNED_JOBS_PER_STEP: int = 1

    __REWARD_WEIGHTS: dict[str, int] = {
        NO_OP_STR: -0.5,
        MACHINE_IDLE_STR: -1,
        MACHINE_UNAVAILABLE_STR: -1,
        JOB_COMPLETED_ON_TIME_STR: 10,
        JOB_COMPLETED_NOT_ON_TIME_STR: 5,
        INVALID_JOB_RECIPE_STR: -5,
        DEADLINE_EXCEEDED_STR: -5,
        ILLEGAL_ACTION_STR: -10,
        NEUTRAL_STR: 0,
    }

    __REWARD_TIME_PENALTIES: dict[str, dict[str, int | float]] = {
        "10_hrs": {"in_s": 36_000, "weight": 0.4},
        "24_hrs": {"in_s": 86_400, "weight": 0.8},
    }

    # observation space constant dict keys
    __PENDING_JOBS_STR: str = "pending_jobs"
    __MACHINES_STR: str = "machines"
    __JOB_PRIORITIES_STR: str = "job_priorities"
    __JOB_REMAINING_TIMES_STR: str = "job_remaining_times"
    __ACHIEVED_GOAL_STR: str = "achieved_goal"
    __DESIRED_GOAL_STR: str = "desired_goal"

    __METADATA: dict[int, str] = {0: "vector", 1: "human"}

    def __init__(
        self, machines: list[Machine], jobs: list[Job], max_steps: int = 10_000
    ) -> None:
        """
        FactoryEnv class constructor method using gym.Space objects for action and observation space
        :param machines: array of Machine instances
        :param jobs: array of Job instances
        :param max_steps: maximum steps for learning in the environment
        """
        super(FactoryEnv, self).__init__()
        self.__total_machines_available: list[Machine] = machines
        self.__machines: list[Machine] = self.__total_machines_available.copy()[
            : self.__MAX_MACHINES
        ]  # restricted len

        self.__jobs: list[Job] = jobs
        self.__pending_jobs: list[Job] = self.__jobs.copy()[
            : self.__BUFFER_LEN
        ]  # buffer of restricted len
        self.__jobs_in_progress: list[tuple[Machine, Job]] = []
        self.__completed_jobs: list[Job] = []
        self.__jobs_completed_per_step_on_time: int = 0
        self.__jobs_completed_per_step_not_on_time: int = 0

        self.__max_steps: int = max_steps
        self.__time_step: int = 0
        self.__step_datetime: str | None = None
        self.__total_factory_process_time: float = 0.0
        self.__render_mode: str = self.__METADATA[1]

        ############
        # callback #
        ############

        self.episode_reward_sum: float = 0.0  # for callback graphing train performance

        ################
        # action space #
        ################

        self.action_space = gym.spaces.Discrete(
            len(self.__machines) * self.__BUFFER_LEN + self.__NO_OP_SPACE
        )

        #####################
        # observation space #
        #####################

        self.__desired_goal: np.ndarray[float] = np.ones(
            shape=self.__NUM_GOALS, dtype=np.float64
        )  # jobs to complete, max machine efficiency
        self.__achieved_goal: np.ndarray[float] = np.zeros(
            shape=self.__NUM_GOALS, dtype=np.float64
        )  # jobs completed on time, machine efficiency
        self.__jobs_priorities: np.ndarray[float] = np.zeros(
            shape=self.__BUFFER_LEN, dtype=np.float64
        )  # job priorities -- should remain constant per buffer
        self.__jobs_not_completed_on_time: int = 0

        pending_jobs_space: gym.spaces.Box = gym.spaces.Box(
            low=0, high=1, shape=(self.__BUFFER_LEN,), dtype=np.float64
        )
        machine_space: gym.spaces.Box = gym.spaces.Box(
            low=0,
            high=1,
            shape=(len(self.__machines) * self.__BUFFER_LEN,),
            dtype=np.float64,
        )
        job_priorities_space: gym.spaces.Box = gym.spaces.Box(
            low=0, high=3, shape=(self.__BUFFER_LEN,), dtype=np.float64
        )
        job_remaining_times_space: gym.spaces.Box = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.__BUFFER_LEN,), dtype=np.float64
        )
        achieved_goal_space: gym.spaces.Box = gym.spaces.Box(
            low=0, high=1, shape=(len(self.__achieved_goal),), dtype=np.float64
        )
        desired_goal_space: gym.spaces.Box = gym.spaces.Box(
            low=0, high=1, shape=(len(self.__desired_goal),), dtype=np.float64
        )

        self.observation_space: gym.spaces.Dict = gym.spaces.Dict(
            {
                self.__PENDING_JOBS_STR: pending_jobs_space,
                self.__MACHINES_STR: machine_space,
                self.__JOB_PRIORITIES_STR: job_priorities_space,
                self.__JOB_REMAINING_TIMES_STR: job_remaining_times_space,
                self.__ACHIEVED_GOAL_STR: achieved_goal_space,
                self.__DESIRED_GOAL_STR: desired_goal_space,
            }
        )
        self.__set_job_priority_observations()

    def __set_job_priority_observations(self):
        """
        Initializes the job priorities for the observation dictionary -- this should remain constant per buffer
        """
        for job in self.__pending_jobs:
            self.__jobs_priorities[job.get_id()] = (
                job.get_priority() / job.MAX_PRIORITY_LEVEL
            )

    def __compute_achieved_goals(self) -> None:
        """
        Calculate achieved tardiness and machine efficiency goals, each bounded [0, 1], for upcoming observation
        Helper private method for Env get_obs() method
        """
        # update achieved tardiness goal observation
        num_jobs_completed_on_time: int = sum(
            [1 for job in self.__completed_jobs if not job.is_past_deadline_date()]
        )
        proportion_jobs_completed_on_time: float = (
            num_jobs_completed_on_time / len(self.__completed_jobs)
            if num_jobs_completed_on_time > 0 and len(self.__completed_jobs) > 0
            else 0.0
        )

        # update achieved machine efficiency goal observation
        total_machine_run_times: float = sum(
            [
                machine.get_time_active() + machine.get_time_idle()
                for machine in self.__machines
            ]
        )
        total_machine_activity_times: float = sum(
            [machine.get_time_active() for machine in self.__machines]
        )
        proportion_machines_efficiency: float = (
            total_machine_activity_times / total_machine_run_times
            if total_machine_run_times > 0 and total_machine_activity_times > 0
            else 0.0
        )

        # update achieved goals state for upcoming observation
        self.__achieved_goal = np.array(
            object=[proportion_jobs_completed_on_time, proportion_machines_efficiency],
            dtype=np.float64,
        )

    def get_obs(self) -> dict[str, np.ndarray[float]]:
        """
        return: obs dict containing binary arrays for pending jobs, machine activity, job priorities, times, and goals
        """
        # update pending jobs observation
        is_pending_jobs: np.ndarray = np.zeros(self.__BUFFER_LEN, dtype=np.float64)
        for job in self.__pending_jobs:
            is_pending_jobs[job.get_id()] = 1.0

        # update machines and active jobs mapping observation
        is_machines_active_jobs: np.ndarray = np.zeros(
            (len(self.__machines), self.__BUFFER_LEN), dtype=np.float64
        )
        for machine in self.__machines:
            for job in machine.get_active_jobs():
                is_machines_active_jobs[machine.get_id(), job.get_id()] = 1.0

        # update job remaining times observation TODO: should we also include times for jobs completed past deadlines?
        job_remaining_times: np.ndarray = np.zeros(self.__BUFFER_LEN, dtype=np.float64)
        current_datetime: datetime = datetime.now()
        for job in [
            *self.__pending_jobs,
            *[job_in_progress[1] for job_in_progress in self.__jobs_in_progress],
        ]:
            job_remaining_times[job.get_id()] = (
                job.get_deadline_datetime() - current_datetime
            ).total_seconds()

        self.__compute_achieved_goals()
        return {
            self.__PENDING_JOBS_STR: is_pending_jobs,
            self.__MACHINES_STR: is_machines_active_jobs.flatten(),
            self.__JOB_PRIORITIES_STR: self.__jobs_priorities,
            self.__JOB_REMAINING_TIMES_STR: job_remaining_times,
            self.__ACHIEVED_GOAL_STR: self.__achieved_goal,
            self.__DESIRED_GOAL_STR: self.__desired_goal,
        }

    def __compute_reward_partial_penalties(self) -> float:
        """
        Calculate partial weights proportional to deadline: 40% if <= 10 hours, 80% <= 24hours, 100% > 24 hours.
        Helper private method for __compute_custom_reward(), which is a helper private method for step()
        :return: sum of reward penalties for each overdue incomplete job proportional to duration past its deadline
        """
        total_time_past_job_deadlines: float = 0.0
        time_past_job_deadline: float

        for job in [
            *self.__pending_jobs,
            *[job_in_progress[1] for job_in_progress in self.__jobs_in_progress],
        ]:
            if job.get_deadline_datetime() < datetime.now():
                job.set_is_past_deadline_date(is_past_deadline_date=True)
                time_past_job_deadline = (
                    datetime.now() - job.get_deadline_datetime()
                ).seconds

                if (
                    time_past_job_deadline
                    <= self.__REWARD_TIME_PENALTIES["10_hrs"]["in_s"]
                ):
                    total_time_past_job_deadlines += (
                        self.__REWARD_TIME_PENALTIES["10_hrs"]["weight"]
                        * time_past_job_deadline
                    )
                elif (
                    self.__REWARD_TIME_PENALTIES["10_hrs"]["in_s"]
                    < time_past_job_deadline
                    <= self.__REWARD_TIME_PENALTIES["24_hrs"]["in_s"]
                ):
                    total_time_past_job_deadlines += (
                        self.__REWARD_TIME_PENALTIES["24_hrs"]["weight"]
                        * time_past_job_deadline
                    )
                else:
                    total_time_past_job_deadlines += time_past_job_deadline

        return (
            self.__REWARD_WEIGHTS[self.DEADLINE_EXCEEDED_STR]
            * total_time_past_job_deadlines
        )

    def __compute_custom_reward(self) -> float:
        """
        Compute step reward based on minimizing of tardiness and maximizing of machine efficiency, and
        increment episode reward sum for callback graphing of the training performance.
        Helper private method for the overridden env step() method
        :return: total sum of proportional rewards for the step, including all proportional penalties
        """
        # init reward with sum of reward for each completed job, on and not on time, since the previous step
        reward: float = (
            self.__REWARD_WEIGHTS[self.JOB_COMPLETED_ON_TIME_STR]
            * self.__jobs_completed_per_step_on_time
            + self.__REWARD_WEIGHTS[self.JOB_COMPLETED_NOT_ON_TIME_STR]
        ) * self.__jobs_completed_per_step_not_on_time
        self.__jobs_completed_per_step_on_time = (
            self.__jobs_completed_per_step_not_on_time
        ) = 0

        for machine in self.__machines:
            if machine.is_available():
                # increment reward penalty proportional sum total machine idle time
                reward += (
                    self.__REWARD_WEIGHTS[self.MACHINE_IDLE_STR]
                    * machine.get_time_idle()
                )
        return reward + self.__compute_reward_partial_penalties()

    def __update_unavailable_machine_state(self, machine: Machine) -> int:
        """
        Checks unavailable machine job processing states and updates on completions since the last step.
        Updates time of progress for each recipe being processed by an unavailable machine.
        Helper private method for __update_factory_env_state(), which is a helper private method for step()
        :param machine: a Machine object that was unavailable for assigning jobs to in the prior step
        :return: the number of job recipes that have been completed by a machine since the last step
        """
        num_recipes_complete: int = 0
        for job in machine.get_active_jobs():
            job_time_diff_seconds = (
                datetime.now() - job.get_start_op_datetime()
            ).seconds

            for recipe in job.get_recipes_in_progress():
                if recipe.get_process_time() <= job_time_diff_seconds:
                    job.set_recipe_completed(completed_recipe=recipe)
                    self.__completed_jobs.append(
                        self.__jobs_in_progress.pop(
                            self.__jobs_in_progress.index((machine, job))
                        )[1]
                    )
                    machine.remove_job_assignment(job=job)
                    num_recipes_complete += 1

                    # increment jobs completed counters based on whether on time or not
                    if job.get_deadline_datetime() < datetime.now():
                        self.__jobs_completed_per_step_on_time += 1
                    else:
                        self.__jobs_completed_per_step_not_on_time += 1
        return num_recipes_complete

    def __is_jobs_done(self) -> bool:
        return len(self.__completed_jobs) == self.__BUFFER_LEN

    def __update_factory_env_state(self) -> tuple[bool, int]:
        """
        Check and update the status of machines and jobs being processed since the last step.
        Updates time in total factory process, and each Machine object's total activity or idleness.
        Helper private method for the overridden env step() method
        :return: conditional for if all jobs in the buffer are completed, number of jobs completed in current step
        """
        num_jobs_complete: int = 0

        # update factory environment total process time
        self.__step_datetime = (
            self.__step_datetime if self.__step_datetime else datetime.now()
        )
        time_diff_seconds: float = (datetime.now() - self.__step_datetime).seconds
        self.__total_factory_process_time += time_diff_seconds
        self.__step_datetime = datetime.now()

        # check and update each machine job processing state, and update activity and idle times
        for machine in self.__machines:
            machine.set_timestamp_status(
                machine.get_timestamp_status()
                if machine.get_timestamp_status()
                else datetime.now()
            )
            time_diff_seconds = (
                datetime.now() - machine.get_timestamp_status()
            ).seconds

            if not machine.is_available():
                num_jobs_complete += self.__update_unavailable_machine_state(
                    machine=machine
                )
                machine.set_time_active(
                    machine.get_time_active() + time_diff_seconds
                )  # update machine active time
            else:
                machine.set_time_idle(
                    machine.get_time_idle() + time_diff_seconds
                )  # update machine idle time
            machine.set_timestamp_status(
                timestamp_status_at_step=datetime.now()
            )  # incrementally set each step - critical

        return self.__is_jobs_done(), num_jobs_complete

    def __init_machine_job(self, selected_machine: Machine, selected_job: Job) -> bool:
        """
        Add one pending job to one available machine given one job recipe is valid for given machine.
        Helper private method for Env step() method
        :param selected_machine: the available machine pending job assignment
        :param selected_job: the available job pending machine assignment
        :return: True if machine is assigned new pending job, otherwise False
        """
        if selected_machine.assign_job(job_to_assign=selected_job):
            self.__jobs_in_progress.append((selected_machine, selected_job))
            self.__pending_jobs.remove(selected_job)
        return not selected_machine.is_available()

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray[any]], float, bool, bool, dict[str, str]]:
        """
        Take a single step in the factory environment.
        TODO: optimise different rewards being returned -- currently just brainstorm placeholders for env testing
        :param action: the agent's action to take in the step
        :return: (observation, reward, terminated, truncated, info)
        """
        print(action)

        (
            is_terminated,
            num_jobs_complete,
        ) = (
            self.__update_factory_env_state()
        )  # check and update env, job and machine status
        step_reward: float = self.__compute_custom_reward()  # compute step reward

        if action == len(self.__machines) * self.__BUFFER_LEN:
            # no operation is returned as the action for the step  # TODO: reward condition if machine available or not?
            self.episode_reward_sum += (
                self.__REWARD_WEIGHTS[self.NO_OP_STR] + step_reward
            )
            return (
                self.get_obs(),  # observation
                self.__REWARD_WEIGHTS[self.NO_OP_STR] + step_reward,  # reward
                is_terminated,  # terminated
                True,  # truncated
                {"Error": self.NO_OP_STR},  # info
            )

        action_selected_machine = self.__machines[
            action // self.__BUFFER_LEN
        ]  # get action selected machine
        if action_selected_machine.is_available():
            self.__time_step += 1  # increment step counter only when the action selected machine is available
            is_terminated = self.__time_step > self.__max_steps

            action_selected_job = self.__jobs[
                action % self.__BUFFER_LEN
            ]  # get action selected job
            if self.__init_machine_job(
                selected_machine=action_selected_machine,
                selected_job=action_selected_job,
            ):
                print(True)
                print(step_reward)
                # action selected machine is available and action selected job is valid for selected machine
                self.episode_reward_sum += step_reward
                return (
                    self.get_obs(),  # observation
                    step_reward,  # reward
                    is_terminated,  # terminated
                    False,  # truncated
                    {},  # info
                )

            # action selected machine is available but action selected job is invalid for selected machine
            self.episode_reward_sum += (
                self.__REWARD_WEIGHTS[self.INVALID_JOB_RECIPE_STR] + step_reward
            )
            return (
                self.get_obs(),  # observation
                self.__REWARD_WEIGHTS[self.INVALID_JOB_RECIPE_STR]
                + step_reward,  # reward
                is_terminated,  # terminated
                False,  # truncated
                {"Error": self.INVALID_JOB_RECIPE_STR},  # info
            )

        # action selected machine is unavailable
        self.episode_reward_sum += (
            self.__REWARD_WEIGHTS[self.MACHINE_UNAVAILABLE_STR] + step_reward
        )
        return (
            self.get_obs(),  # observation
            self.__REWARD_WEIGHTS[self.MACHINE_UNAVAILABLE_STR] + step_reward,  # reward
            is_terminated,  # terminated
            False,  # truncated
            {"Error": self.MACHINE_UNAVAILABLE_STR},  # info
        )

    def render(self):
        """
        Print the current state of the environment at a step
        """
        for machine in self.__machines:
            print(machine)
            print("/-------------------------/")

        if self.__render_mode == self.__METADATA[0]:
            for machine in self.__machines:
                print(machine)
                print("/-------------------------/")
            print()
            print("********************************")
            print()
            for job in self.__jobs:
                print(job)
                print("/-------------------------/")
        elif self.__render_mode == self.__METADATA[1]:
            data = []
            colors = {
                "Machine 0": "rgb(255, 0, 0)",
                "Machine 1": "rgb(170, 14, 200)",
                "Machine 2": (1, 1, 0.2),
            }

            for machine in self.__machines:
                for job in machine.get_active_jobs():
                    start_time_str = machine.get_timestamp_status().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                    finish_time = machine.get_timestamp_status() + timedelta(
                        seconds=job.get_recipes_in_progress()[
                            0
                        ].get_process_time()  # TODO: refactor when more 1 recipe per job
                    )
                    finish_time_str = finish_time.strftime("%Y-%m-%d %H:%M:%S")
                    num_recipes_completed = len(job.get_recipes_completed()) / len(
                        job.get_recipes()
                    )
                    dp = dict(
                        Task="Job " + str(job.get_id()),
                        Start=start_time_str,
                        Finish=finish_time_str,
                        Complete=num_recipes_completed,
                        Resource="Machine ID: " + str(machine.get_id()),
                    )
                    data.append(dp)

            if not len(data) == 0:
                clear_output(wait=True)
                fig = ff.create_gantt(
                    data,
                    colors=colors,
                    index_col="Resource",
                    show_colorbar=True,
                    title="Job-Machine Assignment",
                )
                fig.show()
                sleep(secs=1)  # TODO: remove after dev testing

    def reset(
        self, seed: int = None, options: str = None
    ) -> tuple[dict[str, np.ndarray[any]], dict[str, str]]:
        """
        Reset the environment state
        """
        self.__time_step = 0
        self.__total_factory_process_time = 0.0

        self.__achieved_goal: np.ndarray[float] = np.zeros(
            shape=self.__NUM_GOALS, dtype=np.float64
        )  # obs space achieved goals

        self.episode_reward_sum = 0  # for callback graphing train performance

        for machine in self.__total_machines_available:
            machine.reset()
        self.__machines: list[Machine] = self.__total_machines_available.copy()[
            : self.__MAX_MACHINES
        ]

        for job in self.__jobs:
            job.reset()
        self.__pending_jobs = self.__jobs.copy()[: self.__BUFFER_LEN]
        self.__jobs_in_progress = []
        self.__completed_jobs = []

        return self.get_obs(), {}

    def close(self) -> None:
        """
        Close the environment
        """
        self.reset()
