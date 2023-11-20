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
from collections import defaultdict
from datetime import datetime


class Job:
    """
    Job class
    """

    ####################
    # public constants #
    ####################

    MAX_NUM_RECIPES_PER_JOB: int = 1
    MAX_PRIORITY_LEVEL: int = 3

    #####################
    # private constants #
    #####################

    __STATUS_STR: dict[int, str] = {
        0: "AVAILABLE",
        1: "IN PROGRESS",
        2: "COMPLETED",
        3: "CANCELLED",
        4: "ERROR",
    }

    __PRIORITY_STR: defaultdict[int, str] = defaultdict(lambda: "NOT DEFINED!")
    __PRIORITY_STR.update({1: "NORMAL", 2: "MEDIUM", 3: "HIGH"})

    __DATETIME_FORMAT: str = "%Y-%m-%d %H:%M:%S"

    def __init__(
        self,
        recipes: list[Recipe],
        factory_id: str,
        process_id: int = 0,
        deadline: str = "2023-12-31 23:59:59",
        priority: int = 1,
    ) -> None:
        """
        Job class constructor method
        :param recipes: list of Recipe objects
        :param factory_id: the ID given to the job by the factory for identification
        :param process_id: the ID of the Job in respect to RL algorithm
        :param deadline: the deadline datetime for the Job: YYYY-MM-DD HH:MM:SS
        :param priority: the priority status for the job as determined by the factory: 1=Normal, 2=Medium, 3=High
        """
        self.__id: int = process_id
        self.__factory_id: str = factory_id
        self.__deadline_datetime_str: str = deadline.strip(" ")
        self.__priority: int = priority
        self.__status: int = 0

        self.__recipes: list[Recipe] = recipes
        self.__recipes_pending: list[Recipe] = recipes.copy()[
            : self.MAX_NUM_RECIPES_PER_JOB
        ]  # limit num recipes to max per job
        self.__recipes_in_progress: list[Recipe] = []
        self.__recipes_completed: list[Recipe] = []

        self.__creation_datetime_str: str = datetime.now().strftime(
            self.__DATETIME_FORMAT
        )
        self.__start_op_datetime: datetime | None = None
        self.__is_past_deadline_date: bool = False

    def __get_datetime(self, datetime_str: str) -> datetime:
        return datetime.strptime(datetime_str, self.__DATETIME_FORMAT)

    def get_recipes(self) -> list[Recipe]:
        return self.__recipes

    def get_id(self) -> int:
        return self.__id

    def get_factory_id(self) -> str:
        return self.__factory_id

    def get_deadline_datetime_str(self) -> str:
        return self.__deadline_datetime_str

    def set_deadline_datetime_str(self, new_deadline_datetime_str: str) -> None:
        self.__deadline_datetime_str = new_deadline_datetime_str

    def get_creation_datetime_str(self) -> str:
        return self.__creation_datetime_str

    def set_creation_datetime_str(self, new_creation_datetime_str: str) -> None:
        self.__creation_datetime_str = new_creation_datetime_str

    def get_creation_datetime(self) -> datetime:
        return self.__get_datetime(self.__creation_datetime_str)

    def get_deadline_datetime(self) -> datetime:
        return self.__get_datetime(self.__deadline_datetime_str)

    def get_start_op_datetime(self) -> datetime:
        return self.__start_op_datetime

    def get_status(self) -> int:
        return self.__status

    def get_priority(self) -> int:
        return self.__priority

    def get_priority_str(self) -> str:
        return self.__PRIORITY_STR[self.__priority]

    def is_past_deadline_date(self) -> bool:
        return self.__is_past_deadline_date

    def set_is_past_deadline_date(self, is_past_deadline_date: bool) -> None:
        self.__is_past_deadline_date = is_past_deadline_date

    def update_priority(self, new_priority: int) -> None:
        self.__priority = new_priority

    def update_status(self, new_status: int) -> None:
        self.__status = new_status

    def get_pending_recipes(self) -> list[Recipe]:
        return self.__recipes_pending

    def get_recipes_in_progress(self) -> list[Recipe]:
        return self.__recipes_in_progress

    def get_recipes_completed(self) -> list[Recipe]:
        return self.__recipes_completed

    def get_max_num_recipes(self) -> int:
        return self.MAX_NUM_RECIPES_PER_JOB

    def update_recipes(self, recipes_update: list[Recipe]) -> None:
        self.__recipes = recipes_update
        self.reset()

    def set_recipe_in_progress(self, recipe: Recipe) -> bool:
        if self.can_perform_recipe(recipe=recipe):
            self.__recipes_in_progress.append(
                self.__recipes_pending.pop(self.__recipes_pending.index(recipe))
            )
            self.__start_op_datetime = (
                datetime.now()
            )  # start job timer in datetime format
            self.__status = 1
        else:
            self.__status = 4
        return self.__status == 1

    def set_recipe_completed(self, completed_recipe: Recipe) -> None:
        self.__recipes_completed.append(completed_recipe)
        print(
            f"Recipe {completed_recipe.get_factory_id()} completed for job {self.__factory_id}"
        )
        self.__recipes_in_progress.remove(completed_recipe)
        self.__start_op_datetime = None  # reset job timer

        if self.__recipes_pending:
            self.__status = 0
        else:
            self.__status = 2
        print("\nAfter processing, job status is:", self.__STATUS_STR[self.__status])

    def can_perform_recipe(self, recipe: Recipe) -> bool:
        return recipe in self.__recipes_pending

    def __str__(self) -> str:
        return (
            f"Job ID: {self.__factory_id}"
            f"\nRecipes: {[recipe.get_factory_id() for recipe in self.__recipes]}"
            f"\nQuantity: {len(self.__recipes)}"
            f"\nCreated: {self.__creation_datetime_str}"
            f"\nDeadline: {self.__deadline_datetime_str}"
            f"\nPriority: {self.get_priority_str()}"
            f"\nStatus: {self.__STATUS_STR[self.__status]}"
            f"\nIn Progress: {[recipe.get_id() for recipe in self.__recipes_in_progress]}"
            f"\nCompleted: {100 * len(self.__recipes_completed) / len(self.__recipes)}%"
        )

    def reset(self) -> None:
        self.__status = 0
        self.__recipes_pending = self.__recipes.copy()[: self.MAX_NUM_RECIPES_PER_JOB]
        self.__recipes_in_progress = []
        self.__recipes_completed = []
        self.__start_op_datetime = None
