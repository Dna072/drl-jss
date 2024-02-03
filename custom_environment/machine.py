"""
Custom Machine class for basic concept:
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

from custom_environment.recipe import Recipe
from custom_environment.job import Job
import datetime


class Machine:
    """
    Machine class
    """

    #####################
    # private constants #
    #####################

    __MAX_JOBS_PER_MACHINE: int = 1
    __AVAILABILITY_STR: dict[bool, str] = {False: "UNAVAILABLE", True: "AVAILABLE"}

    def __init__(
        self,
        valid_recipe_types: list[str],
        factory_id: str,
        process_id: int = 0,
        machine_type: str = "A",
        tray_capacity: int = 1000,
        max_recipes_per_process: int = 1,
    ) -> None:
        """
        Machine class constructor method
        :param factory_id: the ID given to the machine by the factory for identification
        :param process_id: the ID of the Machine in respect to RL algorithm
        :param machine_type: the type of machine as determined by the factory (to define which jobs, recipes it accepts)
        :param tray_capacity: the spatial capacity of the machine tray used for each job
        :param valid_recipe_types: list of the types of recipes are valid for being processed by the machine
        :param max_recipes_per_process: the maximum recipes that can be processed by machine in parallel per step
        """
        self.__id: int = process_id
        self.__factory_id: str = factory_id
        self.__machine_type: str = machine_type  # A, B, C, D...
        self.__tray_capacity: int = tray_capacity
        self.__valid_recipe_types: list[str] = valid_recipe_types  # R1, R2, ...
        self.__max_recipes_per_process: int = max_recipes_per_process

        self.__is_available: bool = True
        self.__active_jobs: list[Job] = []
        self.__pending_jobs: list[Job] = []
        self.__timestamp_current_status: datetime = None
        self.__time_active: float = 0.0
        self.__time_idle: float = 0.0
        self._active_recipe: str = ""
        self._pending_tray_capacity: int = tray_capacity
        self._active_tray_capacity: int = tray_capacity

    def get_id(self) -> int:
        return self.__id

    def get_active_recipe(self) -> str:
        return self._active_recipe

    def get_factory_id(self) -> str:
        return self.__factory_id

    def get_machine_type(self) -> str:
        return self.__machine_type

    def get_tray_capacity(self) -> int:
        return self.__tray_capacity

    def get_pending_tray_capacity(self) -> int:
        return self._pending_tray_capacity

    def get_active_tray_capacity(self) -> int:
        return self._active_tray_capacity

    def get_max_recipes_per_process(self) -> int:
        return self.__max_recipes_per_process

    def set_max_recipes_per_process(self, max_recipes_per_process: int) -> None:
        self.__max_recipes_per_process = max_recipes_per_process

    def is_available(self) -> bool:
        return self.__is_available

    def is_available_str(self) -> str:
        return self.__AVAILABILITY_STR[self.__is_available]

    def get_timestamp_status(self) -> datetime:
        return self.__timestamp_current_status

    def set_timestamp_status(self, timestamp_status_at_step: datetime) -> None:
        self.__timestamp_current_status = timestamp_status_at_step

    def get_time_active(self) -> float:
        return self.__time_active

    def get_time_idle(self) -> float:
        return self.__time_idle

    def get_active_jobs(self) -> list[Job]:
        return self.__active_jobs

    def get_active_capacity_utilization(self) -> float:
        return (self.__tray_capacity - self._active_tray_capacity) / self.__tray_capacity

    def get_pending_jobs(self) -> list[Job]:
        return self.__pending_jobs

    def get_max_num_jobs(self) -> int:
        return self.__MAX_JOBS_PER_MACHINE

    def get_valid_recipes(self):
        return self.__valid_recipe_types

    def set_time_active(self, new_time_active: float) -> None:
        self.__time_active = new_time_active

    def set_time_idle(self, new_time_idle: float) -> None:
        self.__time_idle = new_time_idle

    def update_machine_job_recipes(
        self, job: Job, recipes_update: list[Recipe]
    ) -> None:
        self.__active_jobs[self.__active_jobs.index(job)].update_recipes(
            recipes_update=recipes_update
        )

    def update_tray_capacity(self, new_capacity: int) -> None:
        self.__tray_capacity = new_capacity

    def get_job_valid_recipes(self, job: Job) -> list[Recipe]:
        return [
            recipe
            for recipe in job.get_recipes()
            if recipe.get_recipe_type() in self.__valid_recipe_types
        ]

    def get_job_next_recipe(self, job: Job) -> Recipe:
        recipe = None
        if len(job.get_recipes()) > 0:
            if job.get_recipes()[0].get_recipe_type() in self.__valid_recipe_types:
                recipe = job.get_recipes()[0]

        return recipe

    def can_perform_job(self, job: Job) -> bool:
        for recipe in job.get_pending_recipes():
            if recipe.get_recipe_type() in self.__valid_recipe_types:
                return True

        return False

    def can_perform_any_pending_job(self, pending_jobs: list[Job]) -> bool:
        for job in pending_jobs:
            if self.can_perform_job(job):
                return True

        return False
    def start(self) -> bool:
        # To make learning easier, machines cannot be started when unavailable or with no scheduled jobs
        if len(self.__pending_jobs) == 0 or not self.is_available():
            return False

        self.__is_available = False
        self.__active_jobs.extend(self.__pending_jobs)
        self.__pending_jobs.clear()
        self._active_tray_capacity = self._pending_tray_capacity
        self._pending_tray_capacity = self.__tray_capacity

        return True


    def assign_job(self, job_to_assign: Job) -> bool:
        available_valid_recipes: list[Recipe] = self.get_job_valid_recipes(
            job=job_to_assign
        )

        # print(f"For job {job_to_assign.get_factory_id()} machine availa: {self.__is_available} recipe: {len(available_valid_recipes)}")

        if available_valid_recipes and self.__is_available:
            # print(f"Can assign job to machine {self.get_id()}")
            next_valid_recipe_to_process: Recipe = available_valid_recipes[0]
            is_recipe_assigned: bool = job_to_assign.set_recipe_in_progress(
                next_valid_recipe_to_process
            )

            if is_recipe_assigned:
                # print(f"Assigned job to machine ")
                self.__is_available = False
                self.__timestamp_current_status = datetime.datetime.now()
                self.__active_jobs.append(job_to_assign)

                # for job in self.__active_jobs:
                #     print(f'Active job {self.__factory_id}: {job.get_factory_id()}')

                self._active_recipe = next_valid_recipe_to_process
                job_to_assign.set_recipe_in_progress(
                    recipe=next_valid_recipe_to_process
                )
        return not self.__is_available

    def schedule_job(self, job_to_schedule: Job) -> bool:
        # check if machine has available capacity to take the job
        if self._pending_tray_capacity < job_to_schedule.get_tray_capacity():
            return False

        available_valid_recipe: Recipe = self.get_job_next_recipe(
            job=job_to_schedule
        )

        is_recipe_assigned: bool = False

        if available_valid_recipe and self.__is_available:
            # print(f"Can assign job to machine {self.get_id()}")
            next_valid_recipe_to_process: Recipe = available_valid_recipe

            if len(self.__pending_jobs) > 0:
                # check if active recipe matches job recipe
                if not self._active_recipe == next_valid_recipe_to_process.get_factory_id():
                    return False

            is_recipe_assigned = job_to_schedule.set_recipe_in_progress(
                next_valid_recipe_to_process
            )

            if is_recipe_assigned:
                # print(f"Assigned job to machine ")

                self.__timestamp_current_status = datetime.datetime.now()
                self.__pending_jobs.append(job_to_schedule)
                # update machine tray capacity
                self._pending_tray_capacity -= job_to_schedule.get_tray_capacity()

                # for job in self.__active_jobs:
                #     print(f'Active job {self.__factory_id}: {job.get_factory_id()}')
                self._active_recipe = next_valid_recipe_to_process.get_factory_id()

        return is_recipe_assigned

    def remove_job_assignment(self, job: Job) -> None:
        self.__active_jobs.remove(job)
        self._active_tray_capacity += job.get_tray_capacity()
        if not self.__active_jobs:
            self._active_recipe = ""
            self.__is_available = True

    def remove_completed_jobs(self) -> None:
        self.__active_jobs.clear()
        self._active_tray_capacity = self.__tray_capacity
        self._active_recipe = ""
        self.__is_available = True

    def __str__(self) -> str:
        return (
            f"Type: {self.__machine_type}"
            f"\nTray Capacity: {self.__tray_capacity}"
            f" \nStatus: {self.is_available_str()}"
            f"\nCurrent time spent idle: {self.__time_idle}"
            f"\nCurrent time spend active: {self.__time_active}"
            f"\nValid recipes: {self.__valid_recipe_types}"
            f"\nWorking on recipe(s): "
            f"{[job.get_recipes_in_progress() for job in self.__active_jobs]}"
            f" for the following Job(s): {self.__active_jobs}"
        )

    def reset(self) -> None:
        self.__is_available = True
        self.__active_jobs = []
        self.__pending_jobs = []
        self.__timestamp_current_status = None
        self.__time_active = 0.0
        self.__time_idle = 0.0
        self._active_recipe = ""
        self._pending_tray_capacity = self.__tray_capacity
