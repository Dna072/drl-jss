from environment import FactoryEnv
from job import Job
from machine import Machine

class FacEnv2(FactoryEnv):
    def __init__(self, machines: list[Machine], jobs: list[Job], max_steps: int = 10000) -> None:
        super().__init__(machines, jobs, max_steps)

if __name__ == "__main__":
    factory = FacEnv2(machines=[], jobs=[])
    print("Working...")