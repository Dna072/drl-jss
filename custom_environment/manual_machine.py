from custom_environment.machine import Machine

class ManualMachine(Machine):
    """
    ManualMachine class, extends machine and is machine that is startable by agent
    """
    def __init__(self,
                 valid_recipe_types: list[str],
                 factory_id: str,
                 process_id: int = 0,
                 machine_type: str = "A",
                 tray_capacity: int = 10_000,
                 max_recipes_per_process: int = 1,
                 max_jobs: int = 1
                 ) -> None:
        super().__init__(valid_recipe_types, factory_id, process_id, machine_type)