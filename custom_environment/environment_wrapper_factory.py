from custom_environment.job_factory import create_job, get_random_job_deadline
from custom_environment.environment_wrapper import FactoryEnvWrapper
from custom_environment.machine_factory import create_machine
from custom_environment.recipe_factory import create_recipe
from custom_environment.machine import Machine
from custom_environment.recipe import Recipe
from custom_environment.job import Job


def create_factory_env_wrapper(machines: list[Machine], jobs: list[Job]) -> FactoryEnvWrapper:
    """
    Factory function for creating a FactoryEnvWrapper object
    :param machines: list of Machine objects
    :param jobs: list of Job objects
    :return: FactoryEnvWrapper object
    """
    return FactoryEnvWrapper(machines=machines, jobs=jobs)


def init_custom_factory_env_wrapper(is_verbose: bool = False) -> FactoryEnvWrapper:
    """
    Create a custom FactoryEnvWrapper environment for development and testing
    :param is_verbose: print statements if True
    :return: custom FactoryEnvWrapper environment instance with machine, job and recipe objects
    """
    recipe_objects: list[Recipe] = [
        create_recipe(
            factory_id="R1_ID", process_time=1.0, process_id=0, recipe_type="R1"
        ),
        create_recipe(
            factory_id="R2_ID", process_time=2.0, process_id=1, recipe_type="R2"
        ),
    ]

    if is_verbose:
        print("Recipes:")
        for recipe in recipe_objects:
            print(recipe)
            print("-------")

    jobs: list[Job] = [
        create_job(
            recipes=[(recipe_objects[0])],
            factory_id="J1",
            process_id=0,
            deadline=get_random_job_deadline(),
            priority=1,
        ),
        create_job(
            recipes=[(recipe_objects[1])],
            factory_id="J2",
            process_id=1,
            deadline=get_random_job_deadline(),
            priority=2,
        ),
        create_job(
            recipes=[(recipe_objects[0])],
            factory_id="J3",
            process_id=2,
            deadline=get_random_job_deadline(),
            priority=3,
        ),
    ]

    if is_verbose:
        print("\nJobs:")
        for job in jobs:
            print(job)
            print("-------")

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

    if is_verbose:
        print("\nMachines:")
        for machine in machines:
            print(machine)
            print("-------")

    factory_env_wrapper: FactoryEnvWrapper = create_factory_env_wrapper(machines=machines, jobs=jobs)
    return factory_env_wrapper


if __name__ == "__main__":
    from stable_baselines3.common.env_checker import check_env

    custom_factory_env_wrapper: FactoryEnvWrapper = init_custom_factory_env_wrapper(is_verbose=True)
    print("\nCustom environment wrapper check errors:", check_env(custom_factory_env_wrapper))
