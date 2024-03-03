from custom_environment.environment_factory import init_custom_factory_env
from custom_environment.environment import FactoryEnv
from matplotlib import pyplot as plt
from random import randint
import numpy as np


def get_random_action(env: FactoryEnv):
    n_machines = len(env.get_machines())
    action_length = n_machines * env.get_buffer_size()
    action: np.ndarray = np.array(randint(0, action_length))

    # check for machines that have tray capacity full or almost full and start them
    for idx, m in enumerate(env.get_machines()):
        if m.get_pending_tray_capacity() < 30:
            # start the machine
            action = action_length + (idx + 1)  # Current No-Op plus index of machine + 1

    return action

def run_random_agent():
    machines: int = 2
    jobs: int = 3
    tot_reward: int = 0
    j: int = 0
    r_values: list[float] = []
    tr_values: list[int] = []
    steps: list[int] = []
    env: FactoryEnv = init_custom_factory_env(is_verbose=False)
    while 1:  # the environment has the condition to terminate after #max_steps
        action: np.ndarray = np.array(randint(0, machines * jobs))
        # check if any machines are full, if a machine is full start it
        #for idx, m in env.get_machines():

        o, r, te, tr, i = env.step(action)
        tot_reward += r
        r_values.append(r)
        tr_values.append(tot_reward)
        steps.append(j)
        j += 1
        if te:
            break

    plt.scatter(steps, r_values, c="r", marker="o", s=0.5, label="Instant Rewards")
    plt.scatter(steps, tr_values, c="g", marker="x", s=0.5, label="Cumulative Reward")
    plt.legend()
    plt.show()


def episodic_random_agent(n_episodes: int = 10, env_max_steps: int = 10_000):
    """
    Runs a FIFO agent for #n_episodes and returns an array with the total reward
    for each episode
    """
    machines: int = 2
    jobs: int = 3
    ep_reward = []
    ep_tardiness = []
    ep_jobs_ot = []
    ep_jobs_not = []
    for e in range(n_episodes):
        env = init_custom_factory_env(is_verbose=False, max_steps=env_max_steps)
        tot_reward = 0
        while 1:  # the environment has its own termination clauses, so it will trigger the break
            action = get_random_action(env)
            o, r, te, tr, i = env.step(action)
            curr_tardiness = env.get_tardiness_percentage()
            tot_reward += r
            jobs_ot = env.get_jobs_completed_on_time()
            jobs_not = env.get_jobs_completed_not_on_time()
            if te:
                break
        ep_reward.append(tot_reward)
        ep_tardiness.append(curr_tardiness)
        ep_jobs_ot.append(jobs_ot)
        ep_jobs_not.append(jobs_not)
    return ep_reward, ep_tardiness, ep_jobs_ot, ep_jobs_not


# if __name__ == "__main__":
#     mean = np.ones(n_episodes)*np.mean(ep_values)
#     plt.title("Random Agent - 1000 episodes")
#     plt.scatter(
#         range(episodes), ep_values, c="b", marker="o", s=0.5, label="Tot Rewards Episode"
#     )
#     plt.plot(range(episodes), mean, label="Mean Reward "+str(np.round(mean[0], 2)), c='g')
#     plt.legend()
#     plt.show()
