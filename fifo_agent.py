from custom_environment.environment_factory import init_custom_factory_env
from custom_environment.environment import FactoryEnv
from matplotlib import pyplot as plt

# from random import randint
import numpy as np
from custom_environment.utils import (print_observation, print_scheduled_jobs,
                                      print_uncompleted_jobs_buffer,
                                      print_jobs, print_capacity_obs, indices_of_extreme_n)


def get_fifo_action(env: FactoryEnv):
    n_machines = len(env.get_machines())
    no_op_action: int = n_machines * env.get_buffer_size()
    action = no_op_action  # by default no-op
    # First I look for the first created job
    pj = env.get_pending_jobs()
    uc_jobs = env.get_uncompleted_jobs_buffer()
    ct = [j.get_start_time() for j in pj]  # ct = creation times
    uc_jobs_ct = [j.get_start_time() for j in uc_jobs]
    job_indices = indices_of_extreme_n(ct)
    uc_job_index = uc_jobs_ct.index(min(uc_jobs_ct)) if len(uc_jobs_ct) > 0 else -1
    # find a suitable machine
    am = env.get_machines()  # am = all machines
    # check for machines that have tray capacity full or almost full and start them
    for idx, m in enumerate(am):
        if m.get_pending_tray_capacity() < 30:
            # start the machine
            action += (idx + 1) # Current No-Op plus index of machine + 1
            return action

    if uc_job_index == -1 or min(ct) < min(uc_jobs_ct):
        # assign pending job
        for fifo_job_idx in job_indices:
            for idx, m in enumerate(am):
                if m.can_perform_job(pj[fifo_job_idx]) and m.is_available():
                    if m.get_pending_tray_capacity() < pj[fifo_job_idx].get_tray_capacity():
                        # can't schedule job for machine due to tray capacity limitation
                        continue

                    if (m.get_active_recipe_str() != "" and
                            m.get_active_recipe_str() != pj[fifo_job_idx].get_next_pending_recipe().get_factory_id()):
                        # the scheduled job cannot be done, start the machine instead
                        # action += (idx + 1)
                        continue
                    machine_index = idx
                    return machine_index * env.get_buffer_size() + fifo_job_idx
    else:
        for idx, m in enumerate(am):
            if m.can_perform_job(uc_jobs[uc_job_index]) and m.is_available():
                if m.get_pending_tray_capacity() < uc_jobs[uc_job_index].get_tray_capacity():
                    # can't schedule job for machine due to tray capacity limitation
                    continue

                if (m.get_active_recipe_str() != "" and
                        m.get_active_recipe_str() != uc_jobs[uc_job_index].get_next_pending_recipe().get_factory_id()):
                    # the scheduled job cannot be done, start the machine instead
                    action += (idx + 1)
                    break
                machine_index = idx
                action_offset = n_machines * env.get_buffer_size() + n_machines + 1
                action = action_offset + (machine_index * env.get_buffer_size()) + uc_job_index
                break
        # print("Take Action: ", action)
        # If edd job is not schedulable, start any machines that are available
    if action == no_op_action:
        for idx, m in enumerate(am):
            if m.get_pending_tray_capacity() < 40 and m.is_available():
                # start the machine
                action += (idx + 1)  # Current No-Op plus index of machine + 1
                return action
    # If no machine, then i will just send no_op
    return action


def episodic_fifo_agent(n_episodes: int = 10,
                        env_max_steps: int = 10_000,
                        jobs_buffer_size: int = 10,
                        n_recipes: int = 3,
                        jobs_deadline_ratio: float = 0.3,
                        n_machines: int = 4
                        ):
    """
    Runs a FIFO agent for #n_episodes and returns an array with the total reward
    for each episode
    """
    ep_reward = []
    ep_tardiness = []
    ep_jobs_ot = []
    ep_jobs_not = []
    for e in range(n_episodes):
        env = init_custom_factory_env(is_verbose=False, max_steps=env_max_steps,
                                      is_evaluation=True, buffer_size=jobs_buffer_size,
                                      n_recipes=n_recipes, job_deadline_ratio=jobs_deadline_ratio,
                                      n_machines=n_machines)
        tot_reward = 0
        while 1:  # the environment has its own termination clauses, so it will trigger the break
            action = np.array(get_fifo_action(env))
            o, r, te, tr, i = env.step(action)
            tot_reward += r
            curr_tardiness = env.get_tardiness_percentage()
            jobs_ot = env.get_jobs_completed_on_time()
            jobs_not = env.get_jobs_completed_not_on_time()
            if te:
                break
        ep_reward.append(tot_reward)
        ep_tardiness.append(curr_tardiness)
        ep_jobs_ot.append(jobs_ot)
        ep_jobs_not.append(jobs_not)
    return ep_reward, ep_tardiness, ep_jobs_ot, ep_jobs_not


# -------------------- CHECK IF EVERYTHING BELOW THIS LINE CAN BE DELETED --------------
if __name__ == "__main__":
    machines: int = 2
    recipes: int = 2
    jobs: int = 3
    max_steps: int = 100

    j: int = 0
    tot_reward: int = 0

    env: FactoryEnv = init_custom_factory_env(is_verbose=False, n_recipes=recipes, n_machines=machines, buffer_size=jobs)
    obs, info = env.reset()
    #nr_pending_jobs: int = sum(env.get_obs()["pending_jobs"])

    r_values: list[float] = []
    tr_values: list[int] = []
    steps: list[int] = []

    # Running 1 episode
    while 1:
        print_jobs(env)
        print_uncompleted_jobs_buffer(env)
        # print_observation(obs, nr_machines=len(env.get_machines()))
        print_capacity_obs(obs, env)
        action: np.ndarray = np.array(get_fifo_action(env))
        print(f'Action: {action}')
        o, r, te, tr, i = env.step(action)
        print_scheduled_jobs(env)
        print(f"reward: {r}")
        print(f"info: {i}")
        tot_reward += r
        r_values.append(r)
        tr_values.append(tot_reward)
        steps.append(j)
        j += 1

        test = input('Enter anything to continue: ')
        if te:
            break
#
#     # plt.scatter(steps, r_values, c="r", marker="o", s=0.5, label="Instant Rewards")
#     # plt.scatter(steps, tr_values, c="g", marker="x", s=0.5, label="Cumulative Reward")
#     # plt.legend()
#     # plt.show()
#
#     jhjh = input("hit enter to continue with the process....")
#
#     ep_values = []  # array of episode reward
#     episodes: int = 100
#     j = 0
#     eps = []
#     for e in range(episodes):
#         env = init_custom_factory_env(is_verbose=False)
#         tot_reward = 0
#         while (
#             1
#         ):  # the environment has its own termination clauses, so it will trigger the break
#             action = np.array(get_fifo_action(env, jobs))
#             o, r, te, tr, i = env.step(action)
#             tot_reward += r
#             if te:
#                 break
#         ep_values.append(tot_reward)
#
#
#     plt.scatter(
#         range(episodes), ep_values, c="b", marker="o", s=0.5, label="Tot Rewards Episode"
#     )
#     plt.legend()
#     plt.show()
