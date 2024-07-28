# Deep Reinforcement Learning for Industrial Batch Sequencing

Code repository for the _Uppsala University_ Thesis in _Data Science_

#### Contributors

* Derrick Adjei

# Acknowledgement

This thesis repository modifies and extends upon an [existing project of the same title](https://github.com/University-Project-Repos/DataScienceProject) originally developed in collaboration with

* Adam Ross
* Nicolas Pablo Viola

and under close supervision by

* Ayça Özçelikkale
* Pelle Ingvast (industry)

I sincerely thank the original project contributors for their foundational work, which made this thesis possible.

## Modifications

The original project is modified to exclude all 'requirement-specific' contributions to the original default [`main`](https://github.com/Dna072/drl-jss/tree/main) branch following [the 16th squash commit](https://github.com/Dna072/drl-jss/commit/452446d5309e9930363359e25c004be46352466b) from [PR #27](https://github.com/University-Project-Repos/DataScienceProject/pull/27) since merged on November 29th 2023! This is evident in commits [ed017b9](https://github.com/Dna072/drl-jss/commit/ed017b9a7064b084e716968d6fbc9d62cb96c24a) and [08d9ed8](https://github.com/Dna072/drl-jss/commit/08d9ed8532d06b7fd70e8441be78ff0405f6098d) in the new, since renamed default `master` branch as they jump from #27 to #34 on Dec 2nd and #42 (including #41) on Dec 13th, respectively. The links to the PRs are dead in this clone repository with exception to the new ones since its creation starting again from #1, as is evident in the most recent merge commit in the new default `master` branch being from PR #10.  

# Instructions
 
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
