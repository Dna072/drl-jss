# data from https://allisonhorst.github.io/palmerpenguins/

import matplotlib.pyplot as plt
import numpy as np

algorithms = ("Jobs completed on Time", "Jobs completed Not on time")
penguin_means = {
    'DQN': (6, 0),
    'PPO': (3, 2),
    'RAND': (2, 5)
}

x = np.arange(len(algorithms))  # the label locations
width = 0.25  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(layout='constrained')

for attribute, measurement in penguin_means.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute)
    ax.bar_label(rects, padding=3)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Nr of Jobs')
ax.set_title('Job Completion')
ax.set_xticks(x + width, algorithms)
ax.legend(loc='upper left', ncols=3)
# ax.set_ylim(0, 250)

plt.show()