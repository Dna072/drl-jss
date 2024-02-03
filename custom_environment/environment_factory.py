from custom_environment.job_factory import create_job  # , get_random_job_deadline
from custom_environment.machine_factory import create_machine
from custom_environment.recipe_factory import create_recipe
from custom_environment.environment import FactoryEnv
from custom_environment.machine import Machine
from custom_environment.recipe import Recipe
from custom_environment.job import Job


def create_factory_env(machines: list[Machine], jobs: list[Job], max_steps: int = 5000, is_evaluation: bool = False) -> FactoryEnv:
    """
    Factory function for creating a FactoryEnv object
    :param machines: list of Machine objects
    :param jobs: list of Job objects
    :max_steps: max steps the environment will take before termination
    :is_evaluation: Enables/Disables the termination condition is reward<-1000 for learning
    :return: FactoryEnv object
    """
    return FactoryEnv(machines=machines, jobs=jobs, max_steps=max_steps, is_evaluation=is_evaluation)


def init_custom_factory_env(is_verbose: bool = False, max_steps: int = 5000, is_evaluation: bool = False) -> FactoryEnv:
    """
    Create a custom FactoryEnv environment for development and testing
    :param is_verbose: print statements if True
    :return: custom FactoryEnv environment instance with machine, job and recipe objects
    """
    recipe_objects: list[Recipe] = [
        create_recipe(
            factory_id="R1_ID", process_time=30.0, process_id=0, recipe_type="R1",
        ),
        create_recipe(
            factory_id="R2_ID", process_time=300.0, process_id=1, recipe_type="R2"
        ),
    ]

    if is_verbose:
        print("Recipes:")
        for recipe in recipe_objects:
            print(recipe)
            print("-------")

    jobs: list[Job] = [
        create_job(
            recipes=[(recipe_objects[0]), (recipe_objects[1])],
            factory_id="J0",
            process_id=0,
            deadline=1000
            if [(recipe_objects[0])][0].get_recipe_type() == "R1"
            else 1000,
            factory_time=0,
        ),
        create_job(
            recipes=[(recipe_objects[1])],
            factory_id="J1",
            process_id=1,
            deadline=100
            if [(recipe_objects[1])][0].get_recipe_type() == "R1"
            else 1000,
            factory_time=0,
        ),
        create_job(
            recipes=[(recipe_objects[0])],
            factory_id="J2",
            process_id=2,
            deadline=100
            if [(recipe_objects[0])][0].get_recipe_type() == "R1"
            else 1000,
            factory_time=0,
        ),
    ]

    if is_verbose:
        print("\nJobs:")
        for job in jobs:
            print(job)
            print("-------")

    machines: list[Machine] = [
        create_machine(
            factory_id="M0",
            process_id=0,
            machine_type="A",
            tray_capacity=100,
            valid_recipe_types=["R1"],
            max_recipes_per_process=1,
        ),
        create_machine(
            factory_id="M1",
            process_id=1,
            machine_type="AB",
            tray_capacity=100,
            valid_recipe_types=["R1", "R2"],
            max_recipes_per_process=2,
        ),
        create_machine(
            factory_id="M2",
            process_id=2,
            machine_type="A",
            tray_capacity=100,
            valid_recipe_types=["R1"],
            max_recipes_per_process=2,
        ),
    ]

    if is_verbose:
        print("\nMachines:")
        for machine in machines:
            print(machine)
            print("-------")

    factory_env: FactoryEnv = create_factory_env(machines=machines, jobs=jobs, max_steps=max_steps, is_evaluation=is_evaluation)
    return factory_env


if __name__ == "__main__":
    from stable_baselines3.common.env_checker import check_env

    custom_factory_env: FactoryEnv = init_custom_factory_env(is_verbose=True)
    print("\nCustom environment check errors:", check_env(custom_factory_env))
