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
    deadline_ratio = random.choice([0.25, 0.5, 0.75])
    tray_capacity = random.randint(15, 41)
    return Job(
        recipes=recipes,
        factory_id=factory_id,
        process_id=process_id,
        deadline=deadline if deadline > int(corrected_deadline/deadline_ratio) else int(corrected_deadline/deadline_ratio),
        factory_time=factory_time,
        tray_capacity=tray_capacity
    )


