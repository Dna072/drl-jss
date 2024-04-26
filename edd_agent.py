from custom_environment.environment_factory import init_custom_factory_env
from custom_environment.environment import FactoryEnv
from matplotlib import pyplot as plt

# from random import randint
import numpy as np
from custom_environment.utils import (print_observation, print_scheduled_jobs,
                                      print_uncompleted_jobs_buffer,
                                      print_jobs, print_capacity_obs, indices_of_extreme_n)


def get_edd_action(env: FactoryEnv):
    n_machines = len(env.get_machines())
    no_op_action: int = n_machines * env.get_buffer_size()
    action: int = no_op_action  # by default no-op
    # First I look for the first created job
    pj = env.get_pending_jobs()
    uc_jobs = env.get_uncompleted_jobs_buffer()
    ttd = [j.get_steps_to_deadline() for j in pj]  # ttd = times to deadline
    uc_jobs_ttd = [j.get_steps_to_deadline() for j in uc_jobs]
    job_indices = indices_of_extreme_n(ttd)
    uc_job_index = uc_jobs_ttd.index(min(uc_jobs_ttd)) if len(uc_jobs_ttd) > 0 else -1
    # find a suitable machine
    am = env.get_machines()  # am = all machines
    # print_jobs(env)
    # check for machines that have tray capacity full or almost full and start them
    for idx, m in enumerate(am):
        if m.get_pending_tray_capacity() < 1:
            # start the machine
            action += (idx + 1) # Current No-Op plus index of machine + 1
            return action

    if uc_job_index == -1 or min(ttd) < min(uc_jobs_ttd):
        # assign pending job
        for edd_job_idx in job_indices:
            for idx, m in enumerate(am):
                if m.can_perform_job(pj[edd_job_idx]) and m.is_available():
                    if m.get_pending_tray_capacity() < pj[edd_job_idx].get_tray_capacity():
                        # can't schedule job for machine due to tray capacity limitation
                        continue

                    if (m.get_active_recipe_str() != "" and
                            m.get_active_recipe_str() != pj[edd_job_idx].get_next_pending_recipe().get_factory_id()):
                        # the scheduled job cannot be done, start the machine instead
                        # action += (idx + 1)
                        # return action
                        continue

                    machine_index = idx
                    action = machine_index * env.get_buffer_size() + edd_job_idx
                    return action
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
                return action_offset + (machine_index * env.get_buffer_size()) + uc_job_index
    # print("Take Action: ", action)
    # If edd job is not schedulable, start any machines that are available
    if action == no_op_action:
        for idx, m in enumerate(am):
            if m.get_pending_tray_capacity() < 100 and m.is_available():
                # start the machine
                action += (idx + 1)  # Current No-Op plus index of machine + 1
                return action

    return action


def episodic_edd_agent(n_episodes: int = 10,
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
            action = np.array(get_edd_action(env))
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

#-------------------- CHECK IF EVERYTHING BELOW THIS LINE CAN BE DELETED --------------

if __name__ == "__main__":
    #print("Outside")
    machines: int = 5
    recipes: int = 3
    jobs_buffer_size: int = 3
    max_steps: int = 100000

    j: int = 0
    tot_reward: int = 0

    env: FactoryEnv = init_custom_factory_env(is_verbose=False, buffer_size=jobs_buffer_size,
                                              n_recipes=recipes, job_deadline_ratio=0.3, n_machines=machines)
    env.set_termination_reward(-100000)
    #nr_pending_jobs: int = sum(env.get_obs()["pending_jobs"])

    r_values: list[int] = []
    tr_values: list[int] = []
    tardiness: list[float] = []
    steps: list[int] = []

    ep_values = []  # array of episode reward
    episodes: int = 100
    MAX_STEPS = 400000
    eps = []
    avg_returns_per_episode = []
    tardiness_per_episode = []
    job_times_past_deadline = []
    jobs_completed_on_time = []
    jobs_completed_late = []
    avg_tardiness_of_late_jobs_per_episode = []
    returns = []
    current_step = 0
    # env = init_custom_factory_env(is_verbose=False, max_steps=MAX_STEPS)
    env.set_termination_reward(-100000)
    obs, info = env.reset()

    for e in range(episodes):
        tot_reward = 0
        while (
            1
        ):  # the environment has its own termination clauses, so it will trigger the break
            print_jobs(env)
            print_uncompleted_jobs_buffer(env)
            # print_observation(obs, nr_machines=len(env.get_machines()))
            print_capacity_obs(obs, env)
            action = np.array(get_edd_action(env))
            print(f'Action: {action}')
            obs, reward, te, tr, i = env.step(action)

            #print_capacity_obs(obs)

            print_scheduled_jobs(env, buffer_size=jobs_buffer_size)
            print(f"reward: {reward}")
            print(f"info: {i}")

            returns.append(reward)
            current_step += 1
            test = input('Enter anything to continue: ')
            if te:
                print(f"avg return: {np.sum(returns) / steps}")
                avg_returns_per_episode.append(np.sum(returns) / current_step)
                tardiness_per_episode.append(env.get_tardiness_percentage())
                job_times_past_deadline.append(env.get_avg_time_past_deadline())
                jobs_completed_late.append(i['JOBS_NOT_COMPLETED_ON_TIME'])
                jobs_completed_on_time.append(i['JOBS_COMPLETED_ON_TIME'])
                avg_tardiness_of_late_jobs_per_episode.append(i['AVG_TARDINESS_OF_LATE_JOBS'])
                current_step = 0
                returns = []
                obs, info = env.reset()
                env.set_termination_reward(-10000000)
                break

    print(f"Avg jobs completed on time: {sum(jobs_completed_on_time) / len(jobs_completed_on_time)}")
    print(f"Avg jobs completed late: {sum(jobs_completed_late) / len(jobs_completed_late)}")
    print(f"Avg tardiness of late jobs over episodes: "
          f"{sum(avg_tardiness_of_late_jobs_per_episode) / len(avg_tardiness_of_late_jobs_per_episode)}")

#     fig, axs = plt.subplots(3, 1)
#     fig.suptitle("Avg returns per episode")
#     axs[0].plot(avg_returns_per_episode)
#     axs[0].set_title("Rewards plot")
#     axs[0].set(ylabel="Avg returns")
#
#     axs[1].plot(tardiness_per_episode)
#     axs[1].set_title("Tardiness plot")
#     axs[1].set(ylabel="Tardiness percentage")
#
#     axs[2].plot(job_times_past_deadline)
#     axs[2].set_title("Times past deadline")
#     axs[2].set(ylabel="Time")
#
#     for ax in axs.flat:
#         ax.set(xlabel="Episode")
#
#     for ax in axs.flat:
#         ax.label_outer()
#
#     plt.suptitle('Shortest deadline Evaluation')
#     plt.show()
#
#
#     # plt.scatter(
#     #     range(episodes), ep_values, c="b", marker="o", s=0.5, label="Tot Rewards Episode"
#     # )
#     # plt.legend()
#     # plt.show()
#
