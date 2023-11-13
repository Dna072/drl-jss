from custom_environment.job import Job

from datetime import datetime, date, timedelta
import numpy as np
import random


def create_job(prio: int) -> Job:
    job_id = random.randint(0, 10000)
    r: list[str] = [random.choice(["R1", "R2"])]
    q: int = random.randint(1, 10)
    dl: str = str(date.today() + timedelta(days=random.randint(0, 90)))
    return Job(job_id=job_id, recipes=r, quantity=q, deadline=dl, priority=prio)


if __name__ == "__main__":
    # Parameters for the Poisson distribution
    lambda_normal: int = 3  # Per day
    lambda_high: int = 1  # Per day
    jobs: list[Job] = []
    random.seed(int(datetime.now().strftime("%d%m%Y%H%M%S")))

    # Generate a random number of incoming clients following a Poisson distribution
    incoming_normal: int = np.random.poisson(lambda_normal)
    for _ in range(incoming_normal):
        jobs.append(create_job(prio=1))

    incoming_high: int = np.random.poisson(lambda_high)
    for _ in range(incoming_normal):
        jobs.append(create_job(prio=3))

    print("Jobs:")
    for job in jobs:
        print(job)
        print("-------")
