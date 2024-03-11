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
    tray_capacities = [20, 30, 40, 50]
    corrected_deadline = sum([r.get_process_time() for r in recipes])

    return Job(
        recipes=recipes,
        factory_id=factory_id,
        process_id=process_id,
        deadline=deadline if deadline > corrected_deadline/0.3 else corrected_deadline/0.3,
        factory_time=factory_time,
        tray_capacity=random.choice(tray_capacities)
    )


def get_random_job_deadline() -> str:
    return (date.today() + timedelta(days=randint(0, 90))).strftime("%Y-%m-%d %H:%M:%S")


if __name__ == "__main__":
    recipe_objects: list[Recipe] = [
        create_recipe(
            factory_id="R1_ID", process_time=5.0, process_id=0, recipe_type="R1"
        ),
        create_recipe(
            factory_id="R2_ID", process_time=10.0, process_id=1, recipe_type="R2"
        ),
    ]
    jobs: list[Job] = [
        create_job(
            recipes=[(recipe_objects[0])],
            factory_id="J1",
            process_id=0,
            deadline=get_random_job_deadline(),
        ),
        create_job(
            recipes=[(recipe_objects[1])],
            factory_id="J2",
            process_id=1,
            deadline=get_random_job_deadline(),
        ),
        create_job(
            recipes=[(recipe_objects[0])],
            factory_id="J3",
            process_id=2,
            deadline=get_random_job_deadline(),
        ),
    ]

    print("Jobs:")
    for job in jobs:
        print(job)
        print("-------")
