"""
Custom Recipe class for basic concept:
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


class Recipe:
    """
    Recipe class
    """

    def __init__(
        self,
        factory_id: str,
        process_time: float,
        process_id: int = 0,
        recipe_type: str = "R1",
    ):
        """
        Recipe class constructor method
        :param factory_id: the ID given to the recipe by the factory for identification
        :param process_time: the estimated time for the recipe to be completed
        :param process_id: the ID of the Recipe in respect to RL algorithm
        :param recipe_type: the type of the recipe as given by the factory
        """
        self.__factory_id: str = factory_id
        self.__id: int = process_id
        self.__recipe_type: str = recipe_type
        self.__process_time_len: float = process_time

    def add_recipe_type(self, recipe_type: str) -> None:
        self.__recipe_type = recipe_type

    def get_recipe_type(self) -> str:
        return self.__recipe_type

    def get_recipe_type_id(self) -> int:
        return int(self.__recipe_type[1:])

    def add_factory_id(self, factory_id: str) -> None:
        self.__factory_id = factory_id

    def get_factory_id(self) -> str:
        return self.__factory_id

    def add_id(self, process_id: int) -> None:
        self.__id = process_id

    def get_id(self) -> int:
        return self.__id

    def add_process_time(self, process_time: float) -> None:
        self.__process_time_len = process_time

    def get_process_time(self) -> float:
        return self.__process_time_len

    def __str__(self) -> str:
        return (
            f"Recipe ID: {self.__factory_id}"
            f"\nProcess ID: {self.__id}"
            f"\nRecipe type: {self.__recipe_type}"
            f"\nTotal process time: {self.__process_time_len}"
        )
