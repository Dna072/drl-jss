from custom_environment.dispatch_rules.job_factory_dispatch_rules import (
    create_job,
    get_random_job_arrival,
)
from custom_environment.environment import FactoryEnv
from custom_environment.machine import Machine
from custom_environment.recipe import Recipe
from custom_environment.job import Job
from datetime import datetime
import numpy as np


class EnvWrapperDispatchRules(FactoryEnv):
    """
    Factory environment wrapper class for dispatch rules implementation
    Subclass which extends FactoryEnv class, of which furthermore extends gym.Env:
    """

    def __init__(
        self,
        machines: list[Machine],
        jobs: list[Job],
        recipes: list[Recipe],
        max_steps: int = 10_000,
    ):
        """
        FactorEnv wrapper subclass constructor
        :param machines: array of Machine instances
        :param jobs: array of Job instances
        :param recipes: array of Recipe instances
        :param max_steps: maximum steps for learning in the environment
        """
        super().__init__(
            machines, jobs, max_steps
        )  # REQUIRED: init superclass, passing input arguments to instantiate subclass
        self._jobs_counter: int = 0
        self._recipes: list[Recipe] = recipes

    def get_pending_jobs(self):
        return self._pending_jobs

    def get_machines(self):
        return self._machines

    def get_buffer_size(self):
        return self._BUFFER_LEN

    def is_machine_available(self, action) -> bool:
        selected_machine = self._machines[action // self._BUFFER_LEN]

        return selected_machine.is_available()

    def get_obs(self, is_flatten: bool = True) -> dict[str, np.ndarray[float]]:
        """
        return: obs dict containing binary arrays for pending jobs, machine activity, and pending deadline durations
        """
        ###################################
        # update pending jobs observation #
        ###################################
        is_pending_jobs: np.ndarray = np.zeros(self._BUFFER_LEN, dtype=np.float64)
        for job in self._pending_jobs:
            is_pending_jobs[job.get_id()] = 1.0

        ###########################################
        # update recipe(s) types for pending jobs #
        ###########################################
        recipe_types: np.ndarray = np.zeros(
            self._BUFFER_LEN * Job.MAX_NUM_RECIPES_PER_JOB, dtype=np.float64
        )
        for i in range(Job.MAX_NUM_RECIPES_PER_JOB):
            for job in self._pending_jobs:
                recipe_types[job.get_id() * (i + 1)] = (
                    job.get_recipes()[i].get_recipe_type_id()
                    / self.MAX_RECIPES_IN_ENV_SYSTEM
                )

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
        job_remaining_times: np.ndarray = np.zeros(self._BUFFER_LEN, dtype=np.float64)
        current_datetime: datetime = datetime.now()
        for job in [
            *self._pending_jobs,
            *[job_in_progress[1] for job_in_progress in self._jobs_in_progress],
        ]:
            job_remaining_times[job.get_id()] = (
                job.get_deadline_datetime() - current_datetime
            ).total_seconds() - job.get_remaining_process_time()

            # update max and min duration times for normalizing [0, 1]
            if job_remaining_times[job.get_id()] > max_duration:
                max_duration = job_remaining_times[job.get_id()]
            elif job_remaining_times[job.get_id()] < min_duration:
                min_duration = job_remaining_times[job.get_id()]

        # normalize job pending deadline proportional to recipe processing duration times observation
        for job in [
            *self._pending_jobs,
            *[job_in_progress[1] for job_in_progress in self._jobs_in_progress],
        ]:
            job_remaining_times[job.get_id()] = (
                job_remaining_times[job.get_id()] - min_duration
            ) / (max_duration - min_duration)

        ###########################################################
        # return current observation state object for step update #
        ###########################################################
        if is_flatten:
            return {
                self._PENDING_JOBS_STR: is_pending_jobs,
                self._RECIPE_TYPES_STR: recipe_types,
                self._MACHINES_STR: is_machines_active_jobs.flatten(),
                self._JOB_REMAINING_TIMES_STR: job_remaining_times,
            }
        return {
            self._PENDING_JOBS_STR: is_pending_jobs,
            self._RECIPE_TYPES_STR: recipe_types,
            self._MACHINES_STR: is_machines_active_jobs,
            self._JOB_REMAINING_TIMES_STR: job_remaining_times,
        }

    def step(
        self, action: np.ndarray, is_terminated: bool = False
    ) -> tuple[dict[str, np.ndarray[any]], float, bool, bool, dict[str, str]]:
        """
        Take a single step in the factory environment assuming only when machine is available
        :param action: the agent's action to take in the step
        :param is_terminated: conditional for if agent training is terminated
        :return: (observation, reward, terminated, truncated, info)
        """
        self._time_step += 1
        is_terminated = self._time_step > self._max_steps
        print(f"time step: {self._time_step}")

        step_reward: float = self._compute_custom_reward()

        if action == len(self._machines) * self._BUFFER_LEN:
            # no operation is returned as the action for the step
            self.episode_reward_sum += (
                self._REWARD_WEIGHTS[self.NO_OP_STR] + step_reward
            )
            return (
                self.get_obs(),  # observation
                self._REWARD_WEIGHTS[self.NO_OP_STR] + step_reward,  # reward
                is_terminated,  # terminated
                True,  # truncated
                {"Error": self.NO_OP_STR},  # info
            )

        action_selected_machine = self._machines[
            action // self._BUFFER_LEN
        ]  # get action selected machine

        if action_selected_machine.is_available():
            action_selected_job = self._pending_jobs[
                action % len(self._pending_jobs)
            ]  # get action selected job

            if self._init_machine_job(
                selected_machine=action_selected_machine,
                selected_job=action_selected_job,
            ):
                print(True)
                print(step_reward)
                # action selected machine is available and action selected job is valid for selected machine
                self.episode_reward_sum += (
                    step_reward  # for the callback graphing of agent training
                )
                return (
                    self.get_obs(),  # observation
                    step_reward,  # reward
                    is_terminated,  # terminated
                    False,  # truncated
                    {},  # info
                )

            # action selected machine is available but action selected job is invalid for selected machine
            self.episode_reward_sum += (
                self._REWARD_WEIGHTS[self.INVALID_JOB_RECIPE_STR] + step_reward
            )
            return (
                self.get_obs(),  # observation
                self._REWARD_WEIGHTS[self.INVALID_JOB_RECIPE_STR] + step_reward,  # reward
                is_terminated,  # terminated
                False,  # truncated
                {"Error": self.INVALID_JOB_RECIPE_STR},  # info
            )

        # action selected machine is unavailable
        is_terminated = self._update_factory_env_state()  # assume not already executed w/o available machine
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

    def _create_new_jobs(self):
        # Create number of jobs using Poisson Distribution
        import uuid

        lamda_poisson = 3
        incoming_jobs = np.random.poisson(lamda_poisson)

        for _ in range(incoming_jobs):
            recipe = self._recipes[np.random.randint(len(self._recipes))]
            job = create_job(
                recipes=[recipe],
                factory_id=str(uuid.uuid1()),
                process_id=self._jobs_counter % self._BUFFER_LEN,
                arrival=get_random_job_arrival(),
                priority=1,
            )
            self._jobs.append(job)
            self._jobs_counter += 1

    def get_average_machine_idle_time_percentage(self) -> float:
        total_idle_time = 0

        # if self._total_factory_process_time == 0:
        #     return 0

        for machine in self._machines:
            total_idle_time += machine.get_time_idle()

        return total_idle_time / len(
            self._machines
        )  # / self._total_factory_process_time * 100

    def get_average_machine_utilization_time_percentage(self) -> float:
        total_time = 0

        # if self._total_factory_process_time == 0:
        #     return 0

        for machine in self._machines:
            total_time += machine.get_time_active()

        return total_time / len(
            self._machines
        )  # / self._total_factory_process_time * 100
