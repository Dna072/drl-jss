from custom_environment.environment_factory import init_custom_factory_env
from custom_environment.environment import FactoryEnv
from matplotlib import pyplot as plt
from random import randint
import numpy as np

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
while j < max_steps and nr_pending_jobs > 0:
    action: np.ndarray = np.array(randint(0, machines * jobs))
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

tr_values = []
episodes: int = 100
for e in range(episodes):
    j = 0
    env = init_custom_factory_env(is_verbose=False)
    tot_reward = 0
    while j < max_steps and nr_pending_jobs > 0:
        action = np.array(randint(0, 2 ** ((machines * jobs) - 1)))
        o, r, te, tr, i = env.step(action)
        tot_reward += r
        j += 1
    tr_values.append(tot_reward)

plt.scatter(
    range(episodes), tr_values, c="b", marker="o", s=0.5, label="Tot Rewards Episode"
)
plt.legend()
plt.show()
