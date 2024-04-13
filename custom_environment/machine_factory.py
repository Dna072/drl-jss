from custom_environment.machine import Machine


def create_machine(
    factory_id: str,
    process_id: int,
    machine_type: str,
    tray_capacity: int,
    valid_recipe_types: list[str],
    max_recipes_per_process: int,
):
    """
    Factory function for creating a Machine object
    :param factory_id: the ID given to the machine by the factory for identification
    :param process_id: the ID of the Machine in respect to RL algorithm
    :param machine_type: the type of the machine as determined by the factory (to define which jobs, recipes it accepts)
    :param tray_capacity: the spatial capacity of the machine tray used for each job
    :param valid_recipe_types: list of the types of recipes are valid for being processed by the machine
    :param max_recipes_per_process: the maximum recipes that can be processed by machine in parallel per step
    :return: Machine object
    """
    return Machine(
        factory_id=factory_id,
        process_id=process_id,
        machine_type=machine_type,
        tray_capacity=tray_capacity,
        valid_recipe_types=valid_recipe_types,
        max_recipes_per_process=max_recipes_per_process,
    )


if __name__ == "__main__":
    machines: list[Machine] = [
        create_machine(
            factory_id="M1",
            process_id=0,
            machine_type="A",
            tray_capacity=10_000,
            valid_recipe_types=["R1"],
            max_recipes_per_process=1,
        ),
        create_machine(
            factory_id="M3",
            process_id=1,
            machine_type="AB",
            tray_capacity=10_000,
            valid_recipe_types=["R1", "R2"],
            max_recipes_per_process=2,
        ),
    ]

    print("Machines:")
    for machine in machines:
        print(machine)
        print("-------")
