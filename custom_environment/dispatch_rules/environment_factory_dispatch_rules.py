# from custom_environment.dispatch_rules.environment_wrapper_dispatch_rules import (
#     EnvWrapperDispatchRules,
# )
# from custom_environment.dispatch_rules.job_factory_dispatch_rules import (
#     create_job,
#     get_random_job_arrival,
# )
# from custom_environment.dispatch_rules.job_dispatch_rules import Job
# from custom_environment.machine_factory import create_machine
# from custom_environment.recipe_factory import create_recipe
# from custom_environment.machine import Machine
# from custom_environment.recipe import Recipe


# def create_factory_env(
#     machines: list[Machine], jobs: list[Job], recipes: list[Recipe], max_steps:int = 10
# ) -> EnvWrapperDispatchRules:
#     """
#     Factory function for creating a FactoryEnv object
#     :param machines: list of Machine objects
#     :param jobs: list of Job objects
#     :param recipes: list of Recipe objects
#     :return: FactoryEnv object
#     """
#     return EnvWrapperDispatchRules(
#         machines=machines, jobs=jobs, recipes=recipes, max_steps=max_steps
#     )


# def init_custom_factory_env(is_verbose: bool = False, max_steps: int = 10) -> EnvWrapperDispatchRules:
#     """
#     Create a custom FactoryEnv environment for development and testing
#     :param is_verbose: print statements if True
#     :return: custom FactoryEnv environment instance with machine, job and recipe objects
#     """
#     recipe_objects: list[Recipe] = [
#         create_recipe(
#             factory_id="R1_ID", process_time=15.0, process_id=0, recipe_type="R1"
#         ),
#         create_recipe(
#             factory_id="R2_ID", process_time=12.0, process_id=1, recipe_type="R2"
#         ),
#     ]

#     if is_verbose:
#         print("Recipes:")
#         for recipe in recipe_objects:
#             print(recipe)
#             print("-------")

#     jobs: list[Job] = [
#         create_job(
#             recipes=[(recipe_objects[0])],
#             factory_id="J1",
#             process_id=0,
#             arrival=get_random_job_arrival(),
#             # deadline=get_random_job_deadline(),
#             priority=1,
#         ),
#         create_job(
#             recipes=[(recipe_objects[1])],
#             factory_id="J2",
#             process_id=1,
#             arrival=get_random_job_arrival(),
#             # deadline=get_random_job_deadline(),
#             priority=2,
#         ),
#         create_job(
#             recipes=[(recipe_objects[0])],
#             factory_id="J3",
#             process_id=2,
#             arrival=get_random_job_arrival(),
#             # deadline=get_random_job_deadline(),
#             priority=3,
#         ),
#         create_job(
#             recipes=[(recipe_objects[1])],
#             factory_id="J4",
#             process_id=0,
#             arrival=get_random_job_arrival(),
#             # deadline=get_random_job_deadline(),
#             priority=1,
#         ),
#         create_job(
#             recipes=[(recipe_objects[0])],
#             factory_id="J5",
#             process_id=1,
#             arrival=get_random_job_arrival(),
#             # deadline=get_random_job_deadline(),
#             priority=2,
#         ),
#         create_job(
#             recipes=[(recipe_objects[0])],
#             factory_id="J6",
#             process_id=2,
#             arrival=get_random_job_arrival(),
#             # deadline=get_random_job_deadline(),
#             priority=3,
#         ),
#     ]

#     if is_verbose:
#         print("\nJobs:")
#         for job in jobs:
#             print(job)
#             print("-------")

#     machines: list[Machine] = [
#         create_machine(
#             factory_id="M1",
#             process_id=0,
#             machine_type="A",
#             tray_capacity=10_000,
#             valid_recipe_types=["R1"],
#             max_recipes_per_process=1,
#         ),
#         create_machine(
#             factory_id="M3",
#             process_id=1,
#             machine_type="AB",
#             tray_capacity=10_000,
#             valid_recipe_types=["R1", "R2"],
#             max_recipes_per_process=2,
#         ),
#     ]

#     if is_verbose:
#         print("\nMachines:")
#         for machine in machines:
#             print(machine)
#             print("-------")

#     factory_env: EnvWrapperDispatchRules = create_factory_env(
#         machines=machines, jobs=jobs, recipes=recipe_objects, max_steps=max_steps
#     )
#     return factory_env


# if __name__ == "__main__":
#     from stable_baselines3.common.env_checker import check_env

#     custom_factory_env: EnvWrapperDispatchRules = init_custom_factory_env(
#         is_verbose=True
#     )
#     print("\nCustom environment check errors:", check_env(custom_factory_env))
