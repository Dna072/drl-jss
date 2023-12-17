from custom_environment.environment_factory import init_custom_factory_env
from custom_environment.environment import FactoryEnv
from matplotlib import pyplot as plt

# from random import randint
import numpy as np
from custom_environment.utils import print_observation, print_jobs


def get_edd_action(env: FactoryEnv):
    action = 6  # by default no-op
    # First I look for the first created job
    pj = env.get_pending_jobs()
    ttd = [j.get_steps_to_deadline() for j in pj]  # ttd = times to deadline
    job_index = ttd.index(min(ttd))
    # now i look for a suitable machine for it
    am = env.get_machines()  # am = all machines
    # print_jobs(env)
    for i, m in enumerate(am):
        if m.can_perform_job(pj[job_index]) and m.is_available():
            machine_index = i
            action = machine_index * env.get_buffer_size() + job_index
            break
    # print("Take Action: ", action)
    # If no machine, then i will just send no_op
    return action


def episodic_edd_agent(n_episodes: int = 10, env_max_steps: int = 10_000):
    """
    Runs a FIFO agent for #n_episodes and returns an array with the total reward
    for each episode
    """
    ep_reward = []
    ep_tardiness = []
    ep_jobs_ot = []
    ep_jobs_not = []
    for e in range(n_episodes):
        env = init_custom_factory_env(is_verbose=False, max_steps=env_max_steps, is_evaluation=True)
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

# if __name__ == "__main__":
#     print("Outside")
#     machines: int = 2
#     jobs: int = 3
#     max_steps: int = 100000
#
#     j: int = 0
#     tot_reward: int = 0
#
#     env: FactoryEnv = init_custom_factory_env(is_verbose=False)
#     env.set_termination_reward(-100000)
#     #nr_pending_jobs: int = sum(env.get_obs()["pending_jobs"])
#
#     r_values: list[int] = []
#     tr_values: list[int] = []
#     tardiness: list[float] = []
#     steps: list[int] = []
#
#     ep_values = []  # array of episode reward
#     episodes: int = 100
#     MAX_STEPS = 400000
#     eps = []
#     avg_returns_per_episode = []
#     tardiness_per_episode = []
#     job_times_past_deadline = []
#     jobs_completed_on_time = []
#     jobs_completed_late = []
#     avg_tardiness_of_late_jobs_per_episode = []
#     returns = []
#     current_step = 0
#     env = init_custom_factory_env(is_verbose=False, max_steps=MAX_STEPS)
#     env.set_termination_reward(-100000)
#     obs, info = env.reset()
#
#     for e in range(episodes):
#         tot_reward = 0
#         while (
#             1
#         ):  # the environment has its own termination clauses, so it will trigger the break
#             print_jobs(env)
#             print_observation(obs, nr_machines=len(env.get_machines()))
#             action = np.array(get_edd_action(env))
#             obs, reward, te, tr, i = env.step(action)
#
#             print_observation(obs, nr_machines=len(env.get_machines()))
#             print(f"reward: {reward}")
#             print(f"info: {i}")
#
#             returns.append(reward)
#             current_step += 1
#             if te:
#                 print(f"avg return: {np.sum(returns) / steps}")
#                 avg_returns_per_episode.append(np.sum(returns) / current_step)
#                 tardiness_per_episode.append(env.get_tardiness_percentage())
#                 job_times_past_deadline.append(env.get_avg_time_past_deadline())
#                 jobs_completed_late.append(i['JOBS_NOT_COMPLETED_ON_TIME'])
#                 jobs_completed_on_time.append(i['JOBS_COMPLETED_ON_TIME'])
#                 avg_tardiness_of_late_jobs_per_episode.append(i['AVG_TARDINESS_OF_LATE_JOBS'])
#                 current_step = 0
#                 returns = []
#                 obs, info = env.reset()
#                 env.set_termination_reward(-10000000)
#                 break
#
#     print(f"Avg jobs completed on time: {sum(jobs_completed_on_time) / len(jobs_completed_on_time)}")
#     print(f"Avg jobs completed late: {sum(jobs_completed_late) / len(jobs_completed_late)}")
#     print(f"Avg tardiness of late jobs over episodes: "
#           f"{sum(avg_tardiness_of_late_jobs_per_episode) / len(avg_tardiness_of_late_jobs_per_episode)}")
#
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
