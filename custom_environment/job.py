"""
Job class for basic concept:
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
"""


class Job:
    """
    Job class
    """

    def __init__(
        self,
        recipes: list[str],
        job_id: int = 0,
        quantity: int = 0,
        deadline: str = "3000/01/01",
        priority: int = 1,
    ) -> None:
        self.job_id: int = job_id  # String for now, can be numeric
        self.recipes: list[str] = recipes  # [R1, R2, R3, ...]
        self.r_pending: list[str] = recipes.copy()
        self.quantity: int = quantity
        self.deadline: str = deadline  # YYYY/MM/DD
        self.priority: int = priority  # 1=Normal, 2=Medium, 3=High
        self.status: int = 0  # 0=New, 1=In Progress, 2=Completed, 3=Cancelled?, 4=Error? (For internal use)
        self.r_in_progress: list[str] = []
        self.r_completed: list[str] = []

    def get_recipes(self) -> list[str]:
        return self.recipes

    def get_job_id(self) -> int:
        return self.job_id

    def get_quantity(self) -> int:
        return self.quantity

    def get_dead_line(self) -> str:
        return self.deadline

    def get_status(self) -> int:
        return self.status  # might come in handy in the future

    def get_priority(self) -> int:
        return self.priority

    def get_priority_str(self) -> str:
        ref_table: dict = {1: "Normal", 2: "Medium", 3: "High"}
        try:
            return ref_table[self.priority]
        except KeyError:
            return f"Key '{self.priority}' is not defined!"

    def get_status_str(self) -> str:
        ref_table: dict = {
            0: "New",
            1: "In Progress",
            2: "Completed",
            3: "Cancelled",
            4: "Error",
        }
        try:
            return ref_table[self.status]
        except KeyError:
            return f"Key '{self.status}' is not defined!"

    def update_deadline(self, new_deadline) -> None:
        self.deadline = new_deadline

    def update_priority(self, new_priority) -> None:
        self.priority = new_priority

    def update_status(self, new_status) -> None:
        self.status = new_status

    # I didn't add an update recipes because it would add complexity in the future if someone can alter the recipes

    def recipe_in_progress(self, recipe) -> None:
        self.r_pending.remove(recipe)
        self.r_in_progress.append(recipe)
        self.status = 1

    def recipe_completed(self, recipe) -> None:
        self.r_completed.append(recipe)
        self.r_in_progress.remove(recipe)
        if self.r_pending == [] and self.r_in_progress == []:
            self.status = 2

    def __str__(self) -> str:
        return (
            f"Job ID: {self.job_id}"
            f"\nRecipes: {self.recipes}"
            f"\nQuantity: {self.quantity}"
            f"\nDeadline: {self.deadline}"
            f"\nPriority: {self.get_priority_str()}"
            f"\nStatus: {self.get_status_str()}"
            f"\nIn Progress: {self.r_in_progress}"
            f"\nCompleted: {100 * len(self.r_completed) / len(self.recipes)}%"
        )

    def reset(self):
        self.status = 0
        self.r_pending = self.recipes.copy()
        self.r_in_progress = []
        self.r_completed = []
