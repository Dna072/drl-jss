"""
Custom Job class for basic concept:
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

# from datetime import datetime


class Job:
    """
    Job class
    """

    ####################
    # public constants #
    ####################

    MAX_NUM_RECIPES_PER_JOB: int = 5

    #####################
    # private constants #
    #####################

    __STATUS_AVAILABLE_VAL: int = 0
    __STATUS_IN_PROGRESS_VAL: int = 1
    __STATUS_COMPLETED_VAL: int = 2
    __STATUS_CANCELLED_VAL: int = 3
    __STATUS_ERROR_VAL: int = 4

    __STATUS_STR: dict[int, str] = {
        __STATUS_AVAILABLE_VAL: "AVAILABLE",
        __STATUS_IN_PROGRESS_VAL: "IN PROGRESS",
        __STATUS_COMPLETED_VAL: "COMPLETED",
        __STATUS_CANCELLED_VAL: "CANCELLED",
        __STATUS_ERROR_VAL: "ERROR",
    }

    __DATETIME_FORMAT: str = "%Y-%m-%d %H:%M:%S"

    def __init__(
        self,
        recipes: list[Recipe],
        factory_id: str,
        process_id: int = 0,
        deadline: int = 0,
        factory_time: int = 0,
        tray_capacity: int = 30
    ) -> None:
        """
        Job class constructor method
        :param recipes: list of Recipe objects
        :param factory_id: the ID given to the job by the factory for identification
        :param process_id: the ID of the Job in respect to RL algorithm
        :param deadline: the deadline datetime for the Job: YYYY-MM-DD HH:MM:SS
        """
        self.__id: int = process_id
        self.__factory_id: str = factory_id
        self._steps_to_deadline: float = deadline
        self.__status: int = self.__STATUS_AVAILABLE_VAL
        self.__creation_step: int = factory_time

        self.__recipes: list[Recipe] = recipes
        self.__recipes_pending: list[Recipe] = recipes.copy()[
            : self.MAX_NUM_RECIPES_PER_JOB
        ]  # limit num recipes to max per job
        self.__recipes_in_progress: list[Recipe] = []
        self.__recipes_completed: list[Recipe] = []
        # set number of trays required for job
        self.__tray_capacity: int = tray_capacity

        # self.__start_op_datetime: datetime | None = None
        # self.__is_past_deadline: bool = False

        self._steps_to_recipe_complete: int = 0

    def get_recipes(self) -> list[Recipe]:
        return self.__recipes

    def get_remaining_process_time(self) -> float:
        """
        For time being is sufficient with one recipe per job and
        not considering changes in time for recipes already being processed etc.
        """
        return sum([recipe.get_process_time() for recipe in self.__recipes_pending])

    def get_id(self) -> int:
        return self.__id

    def get_factory_id(self) -> str:
        return self.__factory_id

    def get_steps_to_deadline(self) -> float:
        return self._steps_to_deadline

    def get_tray_capacity(self) -> int:
        return self.__tray_capacity

    def update_steps_to_deadline(self, difference):
        self._steps_to_deadline += difference

    def get_start_time(self) -> int:
        return self.__creation_step

    def get_status(self) -> int:
        return self.__status

    def is_past_deadline_date(self) -> bool:
        return self.__is_past_deadline_date

    def set_is_past_deadline_date(self, is_past_deadline_date: bool) -> None:
        self.__is_past_deadline_date = is_past_deadline_date

    def update_status(self, new_status: int) -> None:
        self.__status = new_status

    def get_pending_recipes(self) -> list[Recipe]:
        return self.__recipes_pending

    def get_next_pending_recipe(self) -> Recipe:
        return self.__recipes_pending[0]

    def get_recipes_in_progress(self) -> list[Recipe]:
        return self.__recipes_in_progress

    def get_recipe_in_progress(self) -> Recipe:
        return self.__recipes_in_progress[0]

    def get_recipes_completed(self) -> list[Recipe]:
        return self.__recipes_completed

    def get_max_num_recipes(self) -> int:
        return self.MAX_NUM_RECIPES_PER_JOB

    def is_completed(self):
        return self.__status == self.__STATUS_COMPLETED_VAL

    def update_recipes(self, recipes_update: list[Recipe]) -> None:
        self.__recipes = recipes_update
        self.reset()

    def set_recipe_in_progress(self, recipe: Recipe) -> bool:
        if self.is_next_recipe(recipe=recipe):
            self.__recipes_in_progress.append(
                self.__recipes_pending.pop(0)
            )
            self._steps_to_recipe_complete = recipe.get_process_time()
            self.__status = self.__STATUS_IN_PROGRESS_VAL
        else:
            self.__status = self.__STATUS_ERROR_VAL
        return self.__status == self.__STATUS_IN_PROGRESS_VAL

    def update_steps_to_recipe_complete(self, difference):
        self._steps_to_recipe_complete += difference

    def get_steps_to_recipe_complete(self):
        return self._steps_to_recipe_complete

    def set_recipe_completed(self, completed_recipe: Recipe) -> None:
        self.__recipes_completed.append(completed_recipe)
        # print(f"Recipe {completed_recipe.get_factory_id()} completed for job {self.__factory_id}")
        self.__recipes_in_progress.remove(completed_recipe)
        # self.__start_op_datetime = None  # reset job timer
        self._steps_to_recipe_complete = 0

        if self.__recipes_pending:
            self.__status = self.__STATUS_AVAILABLE_VAL
        else:
            self.__status = self.__STATUS_COMPLETED_VAL
        # print("\nAfter processing, job status is:", self.__STATUS_STR[self.__status])

    def is_next_recipe(self, recipe: Recipe) -> bool:
        if self.__recipes_pending:
            return recipe.get_factory_id() == self.__recipes_pending[0].get_factory_id()
        else:
            return False

    def __str__(self) -> str:
        return (
            f"Job ID: {self.__factory_id}"
            f" Recipes: {[f'{recipe.get_factory_id()} duration: {recipe.get_process_time()}' for recipe in self.__recipes]}"
            f" Quantity: {len(self.__recipes)}"
            f" Created: {self.__creation_step}"
            f" Steps to Deadline: {self.get_steps_to_deadline()}"
            f" Status: {self.__STATUS_STR[self.__status]}"
            f" In Progress: {[recipe.get_id() for recipe in self.__recipes_in_progress]}"
            f" Completed: {100 * len(self.__recipes_completed) / len(self.__recipes)}%"
        )

    def reset(self) -> None:
        self.__status = 0
        self.__recipes_pending = self.__recipes.copy()[: self.MAX_NUM_RECIPES_PER_JOB]
        self.__recipes_in_progress = []
        self.__recipes_completed = []
        # self.__start_op_datetime = None
        self._steps_to_recipe_complete = 0
        self._steps_to_deadline = (
            100 if self.__recipes[0].get_recipe_type() == "R1" else 1000
        )
