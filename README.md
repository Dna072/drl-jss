# Deep Reinforcement Learning for Industrial Batch Sequencing

Code repository for the _Uppsala University_ Thesis in _Data Science_

#### Contributors

* Derrick Adjei
 
### Install

The installation of libraries listed in the `requirements.txt` file are required for running the application.

Run the following to install all required libraries:

```bash
pip install -r requirements.txt
```

## Instructions
The factory environment is defined in directory `./custom_environment`. The environment is a Gymnasium environment.
The logic of the environment can be updated in `./custom_environment/environment.py`. 
Use `./custom_environment/environment_factory.py` to define environment instantiation, the function `create_factory_env`
creates an environment instance.

The model files, i.e. `./custom_environment/job.py`, `./custom_environment/machine.py`, `./custom_environment/recipe.py` contain
the blueprint of each model instance. You can update the behaviour of the specific models in these files. 

All file names with `_factory` provide helper methods for easily creating instances of a model.

Run the `./manual_agent.py` to start a manual agent and test out the environment.

The dispatch rule agents are `edd_agent.py` (Earliest Due Date), `fifo_agent.py` (First-In-First-Out) 
and `heuristic_agent.py` (prioritizes jobs based on process time to deadline ratio).

The RL agents are `dqn_agent.py` (Deep Q-Network), `a2c_agent.py` (Asynchronous Advantage Actor Critic).


## Contribute

### Code style

Code convention best practices is important in collaborative development for consistent readability and maintainability. 
Contributions require being formatted and linted using the `black` and `ruff` libraries to be merged to the `main` branch.

Run the following to format all files:

```bash
black .
```

Run the following to lint all files:

```bash
ruff .
```
