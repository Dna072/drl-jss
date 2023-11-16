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

    INVALID_ACTION_STR: str = "INVALID ACTION"
    NO_OP_STR: str = "NO-OP"
    IDLE_STR: str = "IDLE"
    DEADLINE_STR: str = "DEADLINE"
    COMPLETED_STR: str = "COMPLETED"
    NEUTRAL_STR: str = "NEUTRAL"
    ILLEGAL_STR: str = "ILLEGAL"
    UNAVAILABLE_STR: str = "UNAVAILABLE"

    MAX_RECIPES_IN_ENV_SYSTEM: int = 2

    #####################
    # private constants #
    #####################

    __BUFFER_LEN: int = 3
    __NO_OP_ACTION: int = 0
    __NUM_GOALS = 2  # minimize tardiness, maximize efficiency

    __MAX_MACHINES: int = 2
    __MAX_MACHINES_ASSIGNED_JOBS_PER_STEP: int = 1

    __REWARD_WEIGHTS: dict[str, int] = {
        IDLE_STR: -1,
        DEADLINE_STR: -5,
        COMPLETED_STR: 5,
        UNAVAILABLE_STR: 1,
        NEUTRAL_STR: 0,
        ILLEGAL_STR: -10,
        NO_OP_STR: __NO_OP_ACTION,
    }

    __REWARD_TIME_PENALTIES: dict[str, dict[str, int | float]] = {
        "10_hrs": {"in_s": 36_000, "weight": 0.4},
        "24_hrs": {"in_s": 86_400, "weight": 0.8},
    }

    # observation space constant dict keys
    __PENDING_JOBS_STR: str = "pending_jobs"
    __MACHINES_STR: str = "machines"
    __ACHIEVED_GOAL: str = "achieved_goal"
    __DESIRED_GOAL: str = "desired_goal"

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

        self.__max_steps: int = max_steps
        self.__time_step: int = 0
        self.__step_datetime: str | None = None
        self.__total_factory_process_time: float = 0.0
        self.__render_mode: str = self.__METADATA[1]

        ############
        # callback #
        ############

        self.episode_reward_sum: int = 0  # for callback graphing train performance

        ################
        # action space #
        ################

        self.action_space: gym.spaces.Discrete = gym.spaces.Discrete(
            2
        )  # binary: Op or No-op

        #####################
        # observation space #
        #####################

        self.__desired_goal: np.ndarray[float] = np.ones(
            shape=self.__NUM_GOALS, dtype=np.float64
        )  # jobs to complete, max machine efficiency
        self.__achieved_goal: np.ndarray[float] = np.zeros(
            shape=self.__NUM_GOALS, dtype=np.float64
        )  # jobs completed on time, machine efficiency
        self.__jobs_not_completed_on_time: int = 0

        pending_jobs_space: gym.spaces.Box = gym.spaces.Box(
            low=0, high=1, shape=(len(self.__jobs),), dtype=np.float64
        )
        machine_space: gym.spaces.Box = gym.spaces.Box(
            low=0,
            high=1,
            shape=(len(self.__machines) * len(self.__jobs),),
            dtype=np.float64,
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
                self.__ACHIEVED_GOAL: achieved_goal_space,
                self.__DESIRED_GOAL: desired_goal_space,
            }
        )

    def __compute_achieved_goals(self) -> None:
        """
        Calculate achieved tardiness and machine efficiency goals, each bounded [0, 1], for upcoming observation
        Helper private method for Env get_obs() method
        """
        # update achieved tardiness goal observation
        num_jobs_completed_on_time: int = sum(
            [1 for job in self.__completed_jobs if job.is_past_deadline_date()]
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
        return: observation dict containing binary arrays for pending jobs, machines' active jobs, and goals
        """
        # update pending jobs observation
        is_pending_jobs = np.zeros(len(self.__jobs), dtype=np.float64)
        for job in self.__pending_jobs:
            is_pending_jobs[job.get_id()] = 1.0

        # update machines and active jobs mapping observation
        is_machines_active_jobs = np.zeros(
            (len(self.__machines), len(self.__jobs)), dtype=np.float64
        )
        for machine in self.__machines:
            for job in machine.get_active_jobs():
                is_machines_active_jobs[machine.get_id(), job.get_id()] = 1.0

        self.__compute_achieved_goals()
        return {
            self.__PENDING_JOBS_STR: is_pending_jobs,
            self.__MACHINES_STR: is_machines_active_jobs.flatten(),
            self.__ACHIEVED_GOAL: self.__achieved_goal,
            self.__DESIRED_GOAL: self.__desired_goal,
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

        return self.__REWARD_WEIGHTS[self.DEADLINE_STR] * total_time_past_job_deadlines

    def __compute_custom_reward(self, num_jobs_complete: int = 0) -> int:
        """
        Compute step reward based on minimizing of tardiness and maximizing of machine efficiency, and
        increment episode reward sum for callback graphing of the training performance.
        Helper private method for the overridden env step() method
        :return: total sum of rewards for the step, including all proportional penalties
        """
        # init reward with sum of greatest value for each completed job since the previous step
        reward: float = self.__REWARD_WEIGHTS[self.COMPLETED_STR] * num_jobs_complete

        for machine in self.__machines:
            if machine.is_available():
                # increment reward penalty for every idle machine proportional to total idle time
                reward += self.__REWARD_WEIGHTS[self.IDLE_STR] * machine.get_time_idle()
            else:
                # increment reward for every active machine
                reward += self.__REWARD_WEIGHTS[self.UNAVAILABLE_STR]

        custom_reward: int = int(
            round(reward + self.__compute_reward_partial_penalties())
        )
        self.episode_reward_sum += (
            custom_reward  # for callback graphing training performance
        )
        return custom_reward

    def __update_unavailable_machine_state(self, machine: Machine) -> int:
        """
        Checks unavailable machine job processing states and updates on completions since the last step.
        Updates time of progress for each recipe being processed by an unavailable machine.
        Helper private method for __update_factory_env_state(), which is a helper private method for step()
        TODO: refactor when more than one recipe per job, as although iterates all recipes, handles just one
        :param machine: a Machine object that was unavailable for assigning jobs to in the prior step
        :return: the number of job recipes that have been completed by a machine since the last step
        """
        num_recipes_complete: int = 0
        for job in machine.get_active_jobs():
            job_time_diff_seconds = (
                datetime.now() - job.get_start_timestamp_status()
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
        return num_recipes_complete

    def __is_jobs_done(self) -> bool:
        return len(self.__completed_jobs) == self.__BUFFER_LEN

    def __update_factory_env_state(self) -> tuple[bool, int]:
        """
        Check and update the status of machines and jobs being processed since the last step.
        Updates time in total factory process, and each Machine object's activity or idleness.
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
            )  # incrementally set each step

        return self.__is_jobs_done(), num_jobs_complete

    def __init_machine_job(self, machine: Machine) -> int:
        """
        Add pending job(s) to an available machine given job(s) recipe(s) are valid for given machine.
        Pending jobs are added to machine in order of priority and time remaining until deadline, given valid recipes.
        Helper private method for Env step() method
        :param machine: the available machine pending job assignment
        :return: 1 to represent one machine is assigned a new pending job, otherwise 0 to indicate no jobs are assigned
        """
        num_jobs_assigned: int = 0
        current_date_time: datetime = datetime.now()

        for pending_job in sorted(
            self.__pending_jobs,
            key=lambda j: (
                j.get_priority(),
                current_date_time - j.get_deadline_datetime(),
            ),
            reverse=True,
        ):
            if machine.assign_job(job_to_assign=pending_job):
                self.__jobs_in_progress.append((machine, pending_job))
                self.__pending_jobs.remove(pending_job)
                num_jobs_assigned += 1

            # check the number of jobs assigned to a single machine does not exceed the limit
            if machine.get_max_num_jobs() <= num_jobs_assigned:
                break

        return int(num_jobs_assigned > 0)

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray[any]], float, bool, bool, dict[str, str]]:
        """
        Take a single step in the factory environment.
        Assigns first available machine supporting recipe type the highest priority job with most pending deadline.
        TODO: optimise allocation of jobs to available machines for improved learning, ...
        :param action: the agent's action to take in the step
        :return: (observation, reward, terminated, truncated, info)
        """
        (
            is_terminated,
            num_jobs_complete,
        ) = (
            self.__update_factory_env_state()
        )  # check and update env, job and machine status

        is_machine_available: bool = False
        available_machines_assigned_jobs: int = 0

        for machine in self.__machines:
            if (
                available_machines_assigned_jobs
                <= self.__MAX_MACHINES_ASSIGNED_JOBS_PER_STEP
            ):
                if machine.is_available():
                    self.__time_step += 1  # increment step counter only when there is a machine available
                    is_terminated = self.__time_step > self.__max_steps
                    is_machine_available = True

                    if action == self.__NO_OP_ACTION:
                        return (
                            self.get_obs(),  # observation
                            self.__REWARD_WEIGHTS[self.NO_OP_STR],  # reward
                            is_terminated,  # terminated
                            True,  # truncated
                            {"Error": self.NO_OP_STR},  # info
                        )
                    elif action != self.__NO_OP_ACTION:
                        if not self.__pending_jobs:
                            pass  # TODO: no pending jobs are available, should we update with more?
                        available_machines_assigned_jobs += self.__init_machine_job(
                            machine=machine
                        )

        if is_machine_available:
            return (
                self.get_obs(),  # observation
                self.__compute_custom_reward(
                    num_jobs_complete=num_jobs_complete
                ),  # reward
                is_terminated,  # terminated
                False,  # truncated
                {},  # info
            )

        # TODO: define return when no machine is available, and thus no step
        return (
            self.get_obs(),  # observation
            self.__REWARD_WEIGHTS[self.NEUTRAL_STR],  # reward
            is_terminated,  # terminated
            False,  # truncated
            {"Error": self.INVALID_ACTION_STR},  # info
        )

        # action selected machine is unavailable
        self.episode_reward_sum += (
            self._REWARD_WEIGHTS[self.MACHINE_UNAVAILABLE_STR] + step_reward
        )
        return (
            self.get_obs(),  # observation
            self._REWARD_WEIGHTS[self.MACHINE_UNAVAILABLE_STR] + step_reward,  # reward
            is_terminated,  # terminated
            False,  # truncated
            {"Error": self.MACHINE_UNAVAILABLE_STR},  # info
        )

    def render(self):
        """
        Print the current state of the environment at a step
        """
        for machine in self._machines:
            print(machine)
            print("/-------------------------/")

        if self._render_mode == self._METADATA[0]:
            for machine in self._machines:
                print(machine)
                print("/-------------------------/")
            print()
            print("********************************")
            print()
            for job in self._jobs:
                print(job)
                print("/-------------------------/")
        elif self._render_mode == self._METADATA[1]:
            data = []
            colors = {
                "Machine 0": "rgb(255, 0, 0)",
                "Machine 1": "rgb(170, 14, 200)",
                "Machine 2": (1, 1, 0.2),
            }

            for machine in self._machines:
                for job in machine.get_active_jobs():
                    start_time_str = machine.get_timestamp_status().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                    finish_time = machine.get_timestamp_status() + timedelta(
                        seconds=job.get_recipes_in_progress()[0].get_process_time()
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

        # action selected machine is unavailable
        self.episode_reward_sum += (
            self._REWARD_WEIGHTS[self.MACHINE_UNAVAILABLE_STR] + step_reward
        )
        return (
            self.get_obs(),  # observation
            self._REWARD_WEIGHTS[self.MACHINE_UNAVAILABLE_STR] + step_reward,  # reward
            is_terminated,  # terminated
            False,  # truncated
            {"Error": self.MACHINE_UNAVAILABLE_STR},  # info
        )

    def render(self):
        """
        Print the current state of the environment at a step
        """
        for machine in self._machines:
            print(machine)
            print("/-------------------------/")

        if self._render_mode == self._METADATA[0]:
            for machine in self._machines:
                print(machine)
                print("/-------------------------/")
            print()
            print("********************************")
            print()
            for job in self._jobs:
                print(job)
                print("/-------------------------/")
        elif self._render_mode == self._METADATA[1]:
            data = []
            colors = {
                "Machine 0": "rgb(255, 0, 0)",
                "Machine 1": "rgb(170, 14, 200)",
                "Machine 2": (1, 1, 0.2),
            }

            for machine in self._machines:
                for job in machine.get_active_jobs():
                    start_time_str = machine.get_timestamp_status().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                    finish_time = machine.get_timestamp_status() + timedelta(
                        seconds=job.get_recipes_in_progress()[0].get_process_time()
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

    def close(self) -> None:
        """
        Close the environment
        """
        self.reset()
