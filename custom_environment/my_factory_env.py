import gymnasium as gym
import numpy as np
from custom_environment.environment import FactoryEnv
from custom_environment.job import Job
from custom_environment.machine import Machine

class MyFactoryEnv(FactoryEnv):
    def __init__(self, machines: list[Machine], jobs: list[Job], max_steps: int = 10000) -> None:
        super().__init__(machines, jobs, max_steps)

    def has_machines_available(self) -> bool:
        for machine in self._machines:
            if machine.__is_available:
                return True
        
        return False

