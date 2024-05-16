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
    # Define the bin ranges and their frequencies
    frequency = {}
    if recipe.get_factory_id() == '277-1':
        frequency = {1: 16, 2: 24, 3: 18, 4: 13, 5: 4, 6: 5, 7: 2, 8: 12, 9: 6, 10: 4,
                     12: 4, 13: 27, 14: 5, 15: 2, 16: 4, 18: 1, 19: 3, 20: 1, 22: 1,
                     23: 2, 24: 1, 25: 6, 26: 1, 27: 2, 28: 2, 36: 1}
    elif recipe.get_factory_id() == '277-11':
        frequency = {1: 8, 12: 4, 8: 10, 5: 9, 33: 4, 4: 8, 11: 10, 7: 10, 6: 8, 15: 7,
                     13: 6, 17: 8, 10: 8, 18: 5, 24: 1, 9: 12, 14: 9, 2: 10, 16: 6,
                     20: 2, 22: 5, 27: 1, 21: 3, 25: 1, 19: 4, 23: 2}
    elif recipe.get_factory_id() == '277-12':
        frequency = {1: 19, 8: 29, 3: 15, 10: 26, 13: 17, 6: 30, 5: 33, 9: 23, 20: 5, 15: 9, 7: 25,
                     12: 16, 11: 18, 4: 19, 17: 10, 30: 2, 25: 9, 18: 7, 28: 3, 16: 16, 19: 11,
                     27: 6, 29: 3, 34: 5, 24: 9, 14: 11, 21: 7, 26: 5, 37: 1, 23: 1, 2: 12, 32: 1,
                     35: 2, 22: 7, 38: 1, 40: 1, 31: 3, 36: 2, 33: 1}
    elif recipe.get_factory_id() == '277-13':
        frequency = {5: 41, 6: 66, 31: 4, 21: 23, 8: 40, 10: 34, 27: 5, 18: 19, 14: 36, 4: 30,
                     13: 31, 11: 46, 1: 9, 12: 50, 30: 4, 16: 18, 28: 5, 26: 2, 3: 28, 2: 16,
                     7: 52, 9: 52, 15: 29, 24: 7, 19: 18, 20: 23, 32: 5, 17: 21, 40: 4, 23: 6,
                     22: 11, 25: 6, 34: 3, 33: 2}
    elif recipe.get_factory_id() == '277-14':
        frequency = {1: 9, 2: 8, 3: 4, 4: 8, 5: 9, 6: 21, 7: 18, 8: 20, 9: 27, 10: 32, 11: 24,
                     12: 30, 13: 12, 14: 18, 15: 5, 16: 7, 17: 14, 18: 2, 19: 4, 20: 4, 21: 1,
                     22: 1, 24: 3, 25: 2, 26: 7, 27: 1, 28: 5, 30: 1, 31: 1, 33: 1, 35: 4, 39: 1}
    elif recipe.get_factory_id() == '277-15':
        frequency = {1: 39, 2: 39, 3: 35, 4: 58, 5: 45, 6: 98, 7: 98, 8: 116, 9: 128, 10: 199, 11: 103,
                     12: 66, 13: 86, 14: 201, 15: 68, 16: 76, 17: 108, 18: 11, 19: 20, 20: 12, 21: 22, 22: 23,
                     23: 27, 24: 8, 25: 51, 26: 23, 27: 16, 28: 5, 29: 5, 30: 14, 31: 43, 32: 4, 33: 3, 34: 4,
                     35: 3, 36: 1, 37: 1, 38: 1, 39: 3, 40: 1}
    elif recipe.get_factory_id() == '277-18':
        frequency = {2: 1, 3: 2, 4: 1, 5: 2, 6: 5, 7: 1, 8: 2, 9: 3,
                     10: 3, 11: 5, 12: 1, 13: 2, 14: 1, 15: 1, 18: 1}
    elif recipe.get_factory_id() == '277-3':
        frequency = {17: 1, 18: 1, 19: 1, 20: 1}
    elif recipe.get_factory_id() == '277-5':
        frequency = {1: 25, 2: 31, 3: 19, 4: 16, 5: 6, 6: 13, 7: 10, 8: 10, 9: 19, 10: 12, 11: 31,
                     12: 31, 13: 33, 14: 69, 15: 7, 16: 10, 17: 7, 18: 10, 19: 9, 20: 7, 21: 9,
                     22: 9, 23: 9, 24: 6, 25: 10, 26: 9, 27: 3, 31: 1, 32: 4, 33: 1, 34: 2, 35: 1,
                     36: 2, 37: 3}
    elif recipe.get_factory_id() == '277-6':
        frequency = {1: 12, 2: 13, 3: 7, 4: 22, 5: 26, 6: 23, 7: 19, 8: 40, 9: 28, 10: 15,
                     11: 25, 12: 16, 13: 18, 14: 37, 15: 6, 16: 17, 17: 13, 18: 7, 19: 9, 20: 3,
                     21: 5, 22: 3, 23: 7, 24: 1, 25: 1, 26: 2, 27: 6, 28: 1, 29: 5, 30: 1, 32: 4,
                     33: 2, 34: 3, 36: 4, 37: 15, 38: 6, 39: 2, 40: 2}
    elif recipe.get_factory_id() == '277-61':
        frequency = {1: 20, 2: 21, 3: 23, 4: 20, 5: 24, 6: 15, 7: 16, 8: 8, 9: 12, 10: 3,
                     11: 2, 12: 1, 13: 4, 15: 1, 17: 1, 18: 3, 19: 2, 20: 4, 21: 1,
                     23: 1, 24: 1, 27: 1}
    elif recipe.get_factory_id() == '277-62':
        frequency = {1: 11, 2: 7, 3: 6, 4: 13, 5: 8, 6: 7, 7: 17, 8: 17, 9: 7, 10: 5,
                     11: 7, 12: 8, 13: 1, 14: 3, 15: 1, 16: 3, 17: 6, 18: 2,
                     19: 2, 20: 1, 22: 1, 24: 1, 27: 1}
    elif recipe.get_factory_id() == '277-7':
        frequency = {1: 14, 2: 4, 3: 3, 4: 4, 5: 4, 9: 1, 11: 1, 26: 2, 29: 1, 32: 1, 33: 2}
    elif recipe.get_factory_id() == '277-8':
        frequency = {1: 6, 2: 3, 3: 4, 5: 6, 10: 2, 11: 2,
                     13: 2, 16: 1, 17: 1, 18: 2, 25: 1, 29: 1}

    # Normalize frequencies to get probabilities
    total_frequency = sum(frequency.values())
    probabilities = {num: freq / total_frequency for num, freq in frequency.items()}

    # Generate a single number based on the frequency distribution
    tray_size = random.choices(list(probabilities.keys()), weights=list(probabilities.values()), k=1)[0]

    return tray_size
