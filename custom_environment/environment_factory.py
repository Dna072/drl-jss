from custom_environment.job_factory import create_job  # , get_random_job_deadline
from custom_environment.machine_factory import create_machine
from custom_environment.recipe_factory import create_recipe
from custom_environment.environment import FactoryEnv
from custom_environment.machine import Machine
from custom_environment.recipe import Recipe
from custom_environment.job import Job
import numpy as np


def create_factory_env(machines: list[Machine], jobs: list[Job], recipes: list[Recipe],
                       recipe_probs: list[float], max_steps: int = 5000, is_evaluation: bool = False,
                       jobs_buffer_size: int = 3, job_deadline_ratio: float = 0.3,
                       n_machines: int = 2,
                       machine_tray_capacity: int = 300) -> FactoryEnv:
    """
    Factory function for creating a FactoryEnv object
    :param machines: list of Machine objects
    :param jobs: list of Job objects
    :max_steps: max steps the environment will take before termination
    :is_evaluation: Enables/Disables the termination condition is reward<-1000 for learning
    :return: FactoryEnv object
    """
    return FactoryEnv(machines=machines, jobs=jobs, max_steps=max_steps,
                      is_evaluation=is_evaluation, jobs_buffer_size=jobs_buffer_size,
                      recipes=recipes, job_deadline_ratio=job_deadline_ratio,
                      n_machines=n_machines, machine_tray_capacity=machine_tray_capacity,
                      recipe_probs=recipe_probs)


def init_custom_factory_env(is_verbose: bool = False, max_steps: int = 5000,
                            buffer_size: int = 5, n_recipes: int = 2, n_machines: int = 2,
                            is_evaluation: bool = False, job_deadline_ratio: float = 0.3,
                            machine_tray_capacity: int = 40) -> FactoryEnv:
    """
    Create a custom FactoryEnv environment for development and testing
    @param max_steps: Max steps in the env
    @param buffer_size: Number of jobs in buffer
    @param is_verbose: print statements if True
    @param n_recipes: number of recipes in env
    @param n_machines: number of machines in the env
    @param is_evaluation: flag to determine if we are performing evaluation on the env
    @param job_deadline_ratio: ratio of total job recipes to deadline of job
    @param machine_tray_capacity: Maximum capacity of machines in the env
    :return: custom FactoryEnv environment instance with machine, job and recipe objects
    """
    if n_machines > 10:
        n_machines = 10
    recipe_durations = [30, 150, 200, 250, 300, 350]
    seco_recipes = ["277-1", "277-11", "277-12", "277-13","277-14","277-15","277-18","277-3","277-5","277-6","277-61","277-62","277-7","277-8"]
    seco_recipe_durations = [180, 365, 305, 290, 205, 135, 200, 215, 140, 120, 180, 240, 235, 230]
    seco_recipe_freq = [168,161,433,761,305,1877,32,4,451,439,185,135,37,31]

    if n_recipes > len(seco_recipes):
        n_recipes = len(seco_recipes)

    seco_recipes = seco_recipes[:n_recipes]
    seco_recipe_durations = seco_recipe_durations[:n_recipes]
    seco_recipe_freq = seco_recipe_freq[:n_recipes]
    seco_recipe_dist = [i / sum(seco_recipe_freq) for i in seco_recipe_freq]


    recipe_objects: list[Recipe] = [
        create_recipe(
            factory_id=seco_recipes[i],
            process_time=seco_recipe_durations[i],
            process_id=i,
            recipe_type=seco_recipes[i],
        )
        for i in range(len(seco_recipe_durations))
    ]

    # recipe_objects: list[Recipe] = [
    #     create_recipe(
    #         factory_id=f"R{i}",
    #         process_time=recipe_durations[i % len(recipe_durations)],
    #         process_id=i,
    #         recipe_type=f"R{i}",
    #     )
    #     for i in range(n_recipes)
    # ]



    if is_verbose:
        print("Recipes:")
        for recipe in recipe_objects:
            print(recipe)
            print("-------")

    jobs: list[Job] = [
        create_job(
            recipes=[np.random.choice(np.array(recipe_objects))],
            factory_id=f"J{i}",
            process_id=i,
            deadline=0,
            factory_time=0
        ) for i in range(buffer_size)
    ]

    if is_verbose:
        print("\nJobs:")
        for job in jobs:
            print(job)
            print("-------")

    valid_recipes = [["R0", "R1"], ["R0", "R1"], ["R1", "R2"],
                     ["R1", "R3"], ["R5", "R6"], ["R7","R8"],
                     ["R2", "R3"], ["R4","R5"], ["R8", "R6"]]

    seco_machine_names = ["A", "B", "C", "D", "E","F","G","H","I", "J"]
    seco_valid_recipes = [["274-2","274-3","274-4","274-51","274-7","277-1", "277-2", "277-3", "277-4","277-5","277-6","277-7","277-11","277-12","277-13","277-14","277-15","277-61","277-62"],
                          ["274-2","274-3","274-4","274-7","277-1", "277-2", "277-3", "277-4","277-5","277-6","277-7"],
                          ["274-2","274-3","274-4","274-7","277-1", "277-3", "277-4","277-5","277-6","277-7","277-8","277-11","277-12","277-13","277-14","277-15","277-61","277-62"],
                          ["274-2","274-3","274-4","274-7","277-1", "277-3", "277-4","277-5","277-6","277-7","277-11","277-12","277-13"],
                          ["274-4","277-1","277-5","277-6","277-7","277-11","277-12","277-13"],
                          ["274-4","277-1","277-5","277-6","277-7","277-11","277-12","277-13"],
                          ["274-4","277-1","277-5","277-6","277-7","277-11","277-12","277-13"],
                          ["274-2","274-3","274-4","274-51","274-7","277-1", "277-2", "277-3", "277-4","277-5","277-6","277-7","277-8","277-11","277-12","277-13","277-14","277-15","277-18","277-61","277-62"],
                          ["277-1","277-5","277-11","277-12","277-13", "277-14"],
                          ["274-2", "274-3", "274-4", "274-51", "274-7", "277-1", "277-2", "277-3", "277-4", "277-5",
                           "277-6", "277-7", "277-8", "277-11", "277-12", "277-13", "277-14", "277-15", "277-18",
                           "277-61", "277-62"]
                          ]
    machines: list[Machine] = [
        create_machine(
            factory_id=f"M{i}",
            process_id=i,
            machine_type=f"M{i}",
            tray_capacity=machine_tray_capacity,
            valid_recipe_types=seco_valid_recipes[i % len(seco_valid_recipes)],
            max_recipes_per_process=1,
        )
        for i in range(n_machines)
    ]

    if is_verbose:
        print("\nMachines:")
        for machine in machines:
            print(machine)
            print("-------")

    factory_env: FactoryEnv = create_factory_env(machines=machines, jobs=jobs, max_steps=max_steps, recipes=recipe_objects,
                                                 is_evaluation=is_evaluation, jobs_buffer_size=buffer_size,
                                                 job_deadline_ratio=job_deadline_ratio, n_machines=n_machines,
                                                 machine_tray_capacity=machine_tray_capacity, recipe_probs=seco_recipe_dist)
    return factory_env


if __name__ == "__main__":
    from stable_baselines3.common.env_checker import check_env

    custom_factory_env: FactoryEnv = init_custom_factory_env(is_verbose=True)
    print("\nCustom environment check errors:", check_env(custom_factory_env))
