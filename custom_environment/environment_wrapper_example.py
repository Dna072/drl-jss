import numpy as np

from custom_environment.machine import Machine
from custom_environment.job import Job
from environment import FactoryEnv


class EnvWrapperExample(FactoryEnv):
    """
    Factory environment wrapper class example
    Subclass which extends FactoryEnv class, of which furthermore extends gym.Env:

    ##############    ##############    ###########
    # EnvWrapper # <- # FactoryEnv # <- # gym.Env #
    ##############    ##############    ###########

    Use this as a template:
        - change class name `EnvWrapperExample` to `EnvWrapper<name-of-experiment>`
        - change file name `environment_wrapper_example.py` to `environment_wrapper_<name-of-experiment>.py`
    """

    def __init__(self, machines: list[Machine], jobs: list[Job]):
        """
        Subclass constructor. Required to extend all variables and methods from the superclass and eng.Gym.
        Add new variables, or use/override existing protected variables from extended superclass.
        """
        super().__init__(
            machines, jobs
        )  # REQUIRED: init superclass, passing input arguments to instantiate subclass

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray[any]], float, bool, bool, dict[str, str]]:
        """
        Add new methods or override existing methods either in FactoryEnv or in gym.Env that FactoryEnv extends
        """
        self._compute_reward_partial_penalties()  # example of calling a FactoryEnv method, which can also be overridden
        return {}, 1.0, False, False, {}  # example step method override return
