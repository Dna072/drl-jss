from custom_environment.environment import FactoryEnv
from custom_environment.machine import Machine
from custom_environment.job import Job

import pandas as pd


def pick_up_jobs(self, path) -> None:
    """
    In case we want to take jobs and recipes from csv file
    """
    jobs_df: pd.DataFrame = pd.read_csv(path + "jobs.csv")
    for i, r in jobs_df.iterrows():
        r_rec: pd.Series = r["Recipes"].replace("'", "")
        r_rec = r_rec.split(",")
        j = Job(
            recipes=list(r_rec),
            job_id=r["Id"],
            quantity=r["Quantity"],
            deadline=r["Deadline"],
            priority=r["Priority"],
        )
        self.jobs.append(j)


def pick_up_machines(self, path) -> None:
    """
    In case we want to take machines from csv file
    """
    machines_df: pd.DataFrame = pd.read_csv(path + "machines.csv")
    for i, m in machines_df.iterrows():
        m_rec: pd.Series = m["Recipes"].replace("'", "")
        m_rec = m_rec.split(",")
        m = Machine(m_type=m["Type"], k_recipes=list(m_rec), cap=m["Capacity"])
        self.machines.append(m)


def init_custom_factory_env(is_verbose: bool = False) -> FactoryEnv:
    """
    Create a custom FactoryEnv environment for development and testing
    :param is_verbose: print statements if True
    :return: custom FactoryEnv environment instance
    """
    machine_one: Machine = Machine(
        k_recipes=["R1", "R2"], machine_id=0, m_type="A", cap=10_000
    )
    machine_two: Machine = Machine(
        k_recipes=["R1", "R2"], machine_id=1, m_type="A", cap=10_000
    )
    machine_three: Machine = Machine(
        k_recipes=["R2"], machine_id=2, m_type="A", cap=10_000
    )

    if is_verbose:
        print(machine_one)
        print(machine_two)
        print(machine_three)
        print()

    job_one: Job = Job(
        recipes=[1],
        job_id=0,
        quantity=3,
        deadline="2024/01/04",
        priority=1,
    )
    job_two: Job = Job(
        recipes=[2],
        job_id=1,
        quantity=10,
        deadline="2023/10/28",
        priority=2,
    )
    job_three: Job = Job(
        recipes=[3], job_id=2, quantity=5, deadline="2023/12/04", priority=3
    )

    if is_verbose:
        # job_one.recipe_in_progress("A1")
        # job_one.recipe_completed("A1")

        print(job_one)
        print("/--------/")
        print(job_two)
        print("/--------/")
        print(job_three)

        job_one.reset()

    factory_env: FactoryEnv = FactoryEnv(
        machines=[machine_one, machine_two, machine_three],
        jobs=[job_one, job_two, job_three],
    )

    print(factory_env.get_legal_actions())
    return factory_env


if __name__ == "__main__":
    from stable_baselines3.common.env_checker import check_env

    custom_factory_env: FactoryEnv = init_custom_factory_env(is_verbose=True)
    print("\nCustom environment check errors:", check_env(custom_factory_env))
