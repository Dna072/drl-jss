from custom_environment.job import Job
from custom_environment.recipe import Recipe
from custom_environment.recipe_factory import create_recipe

from datetime import date, datetime, timedelta
from random import randint


def create_job(
    recipes: list[Recipe],
    factory_id: str,
    process_id: int,
    arrival: str,
    #deadline: str,
    priority: int,
) -> Job:
    """
    Factory function for creating a Job object
    :param recipes: list of Recipe objects
    :param factory_id: the ID given to the job by the factory for identification
    :param process_id: the ID of the Job in respect to RL algorithm
    :param deadline: the deadline datetime for the Job as determined by the factory: YYYY-MM-DD
    :return: Job object
    """
    return Job(
        recipes=recipes,
        factory_id=factory_id,
        process_id=process_id,
        arrival=arrival,
        #deadline=deadline,
        priority=priority,
    )


def get_random_job_arrival() -> str:
    return (date.today() + timedelta(hours=randint(0, 90))).strftime("%Y-%m-%d %H:%M:%S")

def get_random_job_deadline(arrival: datetime) -> str:
    return (arrival + timedelta(days=randint(0, 90))).strftime("%Y-%m-%d %H:%M:%S")

if __name__ == "__main__":
    recipe_objects: list[Recipe] = [
        create_recipe(
            factory_id="R1_ID", process_time=1.0, process_id=0, recipe_type="R1"
        ),
        create_recipe(
            factory_id="R2_ID", process_time=2.0, process_id=1, recipe_type="R2"
        ),
    ]
    jobs: list[Job] = [
        create_job(
            recipes=[(recipe_objects[0])],
            factory_id="J1",
            process_id=0,
            arrival=get_random_job_arrival(),
            #deadline=get_random_job_deadline(),
            priority=1,
        ),
        create_job(
            recipes=[(recipe_objects[1])],
            factory_id="J2",
            process_id=1,
            arrival=get_random_job_arrival(),
            #deadline=get_random_job_deadline(),
            priority=2,
        ),
        create_job(
            recipes=[(recipe_objects[0])],
            factory_id="J3",
            process_id=2,
            arrival=get_random_job_arrival(),
            #deadline=get_random_job_deadline(),
            priority=3,
        ),
         create_job(
            recipes=[(recipe_objects[1])],
            factory_id="J4",
            process_id=2,
            arrival=get_random_job_arrival(),
            #deadline=get_random_job_deadline(),
            priority=1,
        ),
         create_job(
            recipes=[(recipe_objects[0])],
            factory_id="J5",
            process_id=2,
            arrival=get_random_job_arrival(),
            #deadline=get_random_job_deadline(),
            priority=2,
        ),
    ]

    print("Jobs:")
    for job in jobs:
        print(job)
        print("-------")
