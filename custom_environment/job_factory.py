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
    corrected_deadline = sum([r.get_process_time() for r in recipes])
    deadline_ratio = random.uniform(0.2, 0.28)
    tray_capacity = get_tray_capacity(recipes[0])

    return Job(
        recipes=recipes,
        factory_id=factory_id,
        process_id=process_id,
        deadline=deadline if deadline > int(corrected_deadline / deadline_ratio) else int(
            corrected_deadline / deadline_ratio),
        factory_time=factory_time,
        tray_capacity=tray_capacity
    )


def get_tray_capacity(recipe: Recipe):
    # Generate random weights for each possible tray size from 1 to 40
    tray_sizes = list(range(1, 41))
    random_weights = [random.random() for _ in tray_sizes]

    # Normalize weights to sum up to 1 to create a probability distribution
    total_weight = sum(random_weights)
    probabilities = [weight / total_weight for weight in random_weights]

    # Generate a single number based on the random probability distribution
    tray_size = random.choices(tray_sizes, weights=probabilities, k=1)[0]

    return tray_size