# # data from https://allisonhorst.github.io/palmerpenguins/
#
# import matplotlib.pyplot as plt
# import numpy as np
#
# algorithms = ("Jobs completed on Time", "Jobs completed Not on time")
# penguin_means = {
#     'DQN': (6, 0),
#     'PPO': (3, 2),
#     'RAND': (2, 5)
# }
#
# x = np.arange(len(algorithms))  # the label locations
# width = 0.25  # the width of the bars
# multiplier = 0
#
# fig, ax = plt.subplots(layout='constrained')
#
# for attribute, measurement in penguin_means.items():
#     offset = width * multiplier
#     rects = ax.bar(x + offset, measurement, width, label=attribute)
#     ax.bar_label(rects, padding=3)
#     multiplier += 1
#
# # Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_ylabel('Nr of Jobs')
# ax.set_title('Job Completion')
# ax.set_xticks(x + width, algorithms)
# ax.legend(loc='upper left', ncols=3)
# # ax.set_ylim(0, 250)
#
# plt.show()
import numpy as np


def create_mean_array(input_array, group_size=10):
    # Reshape the array into a 2D array with the specified group size
    reshaped_array = np.reshape(input_array, (len(input_array) // group_size, group_size))

    # Calculate the mean along the second axis (axis=1)
    mean_array = np.mean(reshaped_array, axis=1)

    return mean_array


# Example usage with an array of 100 values
input_array = np.arange(100)
result_array = create_mean_array(input_array, group_size=10)

print("Original array:")
print(input_array)

print("\nArray with the mean of every 10 values:")
print(result_array)
