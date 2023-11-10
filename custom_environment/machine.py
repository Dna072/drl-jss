"""
Machine class for basic concept:
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

from custom_environment.job import Job
import datetime

class Machine:
    """
    Machine class
    """

    def __init__(
        self,
        k_recipes: list[int],
        machine_id: int = 0,
        m_type: str = "A",
        cap: int = 10_000,
    ) -> None:
        self.machine_id: int = machine_id
        self.machine_type: str = m_type  # A, B, C, D...
        self.known_recipes: list[int] = k_recipes  # R1, R2, R3...
        self.tray_capacity: int = cap
        self.status: int = 0  # 0=Free, 1=Busy - assuming you cannot open a machine that is working to add more trays
        self.active_recipe: str = ""
        self.active_jobs: list[Job] = []
        self.timestamp_current_status = datetime.datetime.now()
        self.recipe_times: dict[str, np.float32]  = {"R1": 1.0, "R2":2.5}  #minutes:hours

    def get_known_recipes(self) -> list[int]:
        return self.known_recipes

    def get_machine_id(self) -> int:
        return self.machine_id

    def get_machine_type(self) -> str:
        return self.machine_type

    def get_tray_capacity(self) -> int:
        return self.tray_capacity

    def get_status(self) -> int:
        return self.status

    def get_status_str(self) -> str:
        if self.status == 0:
            return "Free"
        else:
            return "Busy"

    def get_timestamp_status(self) -> datetime:
        return self.timestamp_current_status
        
    def get_active_recipe(self) -> str:
        return self.active_recipe

    def get_active_jobs(self) -> list[Job]:
        return self.active_jobs

    def update_known_recipes(self, new_recipes) -> None:
        self.known_recipes = new_recipes

    def update_tray_capacity(self, new_capacity) -> None:
        self.tray_capacity = new_capacity

    def assign_jobs(self, jobs) -> bool:
#         print("Assign jobs:",jobs)
        #find compatible recipe between jobs and machine: recipe=js[0].get_recipes()
        recipe = (jobs[0].get_recipes())[0]
#         print(jobs,recipe)
        if self.get_status()==0:
            is_assigned: bool = self.set_active_recipe(recipe)        
            if is_assigned:
                self.status = 1
                self.timestamp_current_status = datetime.datetime.now()
                for j in jobs:
                    self.active_jobs.append(j)
                    j.recipe_in_progress(recipe)
            return is_assigned
        else:
            return False #machine is busy!

    def set_active_recipe(self, recipe) -> bool:
        if recipe in self.known_recipes:
            self.active_recipe = recipe
            return True
        return False
    
    def can_perform_recipe(self, recipe) -> bool:
        if recipe in self.known_recipes:
            return True
        return False

    def __str__(self) -> str:
        return (
            f"Type: {self.machine_type}"
            f"\nKnown Recipes: {self.known_recipes}"
            f"\nTray Capacity: {self.tray_capacity}"
            f" \nStatus: {self.get_status_str()}"
            f"\nWorking on recipe: {self.active_recipe} for the following Jobs: {self.active_jobs}"
        )

    def reset(self):
        self.status = 0
        self.active_jobs = []
        self.active_recipe = ""