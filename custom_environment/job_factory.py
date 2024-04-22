import random

from custom_environment.job import Job
from custom_environment.recipe import Recipe
from custom_environment.recipe_factory import create_recipe

from datetime import date, timedelta
from random import randint


def create_job(
    recipes: list[Recipe],
    factory_id: str,
    process_id: int,
    deadline: int,
    factory_time: int,
) -> Job:
    """
    Factory function for creating a Job object
    :param recipes: list of Recipe objects
    :param factory_id: the ID given to the job by the factory for identification
    :param process_id: the ID of the Job in respect to RL algorithm
    :param deadline: the deadline datetime for the Job as determined by the factory: YYYY-MM-DD
    :param factory_time: current factory time
    :return: Job object
    """
    seco_recipes = ["274-2","274-3","274-4","274-51","274-7","277-1", "277-2", "277-3", "277-4","277-5","277-6","277-7","277-8","277-11","277-12","277-13","277-14","277-15","277-18","277-61","277-62"]
    seco_recipe_durations = [155, 280, 210, 120, 175, 180, 75, 215, 140, 140, 120, 235, 230, 365, 305, 290, 205, 135, 200, 180, 240]
    seco_recipe_freq = [0,0,0,0,0,168,161,433,761,305,1877,32,0,4,0,451,439,185,135,37,31]
    seco_recipe_dist = [i/sum(seco_recipe_freq) for i in seco_recipe_freq]


    corrected_deadline = sum([r.get_process_time() for r in recipes])
    deadline_ratio = random.uniform(0.2, 0.41)
    tray_capacity = get_tray_capacity()


    return Job(
        recipes=recipes,
        factory_id=factory_id,
        process_id=process_id,
        deadline=deadline if deadline > int(corrected_deadline/deadline_ratio) else int(corrected_deadline/deadline_ratio),
        factory_time=factory_time,
        tray_capacity=tray_capacity
    )

def get_tray_capacity():
    tray_probs = [15/41, 13/41, 13/41]
    # Generate a random number between 0 and 1
    rand_num = random.random()
    tray_size = 0
    # Determine which range the random integer falls into based on the probabilities
    if rand_num < tray_probs[0]:
        # Range 1 to 15
        tray_size = random.randint(1, 15)
    elif rand_num < tray_probs[0] + tray_probs[1]:
        # Range 16 to 29
        tray_size = random.randint(16, 29)
    else:
        # Range 30 to 40
        tray_size = random.randint(30, 40)

    return tray_size


