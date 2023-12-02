from custom_environment.environment_factory import init_custom_factory_env
from custom_environment.environment import FactoryEnv
from matplotlib import pyplot as plt
from random import randint
import numpy as np
from custom_environment.utils import print_observation, print_jobs

def get_edd_action(env:FactoryEnv):
    job_index = -1
    machine_index = -1
    action = 6 #by default no-op
    #First I look for the first created job
    pj = env.get_pending_jobs()
    ttd = [j.get_steps_to_deadline() for j in pj] #ttd = times to deadline
    print(ttd)
    job_index = ttd.index(min(ttd))
    #now i look for a suitable machine for it
    am = env.get_machines() #am = all machines
    print_jobs(env)

    for i,m in enumerate(am):
        if m.can_perform_job(pj[job_index]) and m.is_available():
            machine_index = i
            action = machine_index * env.get_buffer_size() + job_index

            break
    print("Take Action: ",action)
    #hjhj = input("Enter")
    #If no machine, then i will just send no_op
    return action

machines: int = 2
jobs: int = 3
max_steps: int = 100

j: int = 0
tot_reward: int = 0

env: FactoryEnv = init_custom_factory_env(is_verbose=False)
nr_pending_jobs: int = sum(env.get_obs()["pending_jobs"])

r_values: list[int] = []
tr_values: list[int] = []
steps: list[int] = []

# Running 1 episode
while 1:
    action: np.ndarray = np.array(get_edd_action(env))
    o, r, te, tr, i = env.step(action)
    print_observation(obs=o, nr_machines=machines)
    print(f'reward: {r}')
    tot_reward += r
    r_values.append(r)
    tr_values.append(tot_reward)
    steps.append(j)
    j+=1
    if te:
        break

plt.scatter(steps, r_values, c="r", marker="o", s=0.5, label="Instant Rewards")
plt.scatter(steps, tr_values, c="g", marker="x", s=0.5, label="Cumulative Reward")
plt.legend()
plt.show()

jhjh = input("hit enter to continue with the process....")

ep_values = [] #array of episode reward
episodes: int = 100
j = 0
eps = []
for e in range(episodes):
    env = init_custom_factory_env(is_verbose=False)
    tot_reward = 0
    while 1 : #the environment has its own termination clauses, so it will trigger the break
        action = np.array(get_edd_action(env,jobs))
        o, r, te, tr, i = env.step(action)
        tot_reward += r
        if te:
            break
    ep_values.append(tot_reward)



plt.scatter(range(episodes), ep_values, c="b", marker="o", s=0.5, label="Tot Rewards Episode")
plt.legend()
plt.show()
