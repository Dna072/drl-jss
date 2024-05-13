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
    # Define the bin ranges and their frequencies
    bin_ranges = [
        (1.00, 6.50), (6.50, 12.00), (12.00, 17.50),
        (17.50, 23.00), (23.00, 28.50), (28.50, 34.00),
        (34.00, 40)
    ]
    frequencies = [1261, 1507, 1341, 349, 293, 124, 71]

    # Generate a random bin based on the frequencies
    random_bin = random.choices(bin_ranges, weights=frequencies)[0]

    # Generate a random integer within the chosen bin range
    tray_size = random.randint(int(random_bin[0]), int(random_bin[1]))

    return tray_size


