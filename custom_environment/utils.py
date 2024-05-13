#from custom_environment.environment import FactoryEnv
import numpy as np
import pickle
import heapq

class TextColors:
    RESET = "\033[0m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"


def indices_of_extreme_n(arr, find_minimum=True):
    """
    @param arr: Array to get indices of min/max n elements
    @param find_minimum: Flag to indicate whether to return min or max elements indices
    """
    # Create a heap based on the desired extremum
    n = len(arr)
    if find_minimum:
        heap = [(val, idx) for idx, val in enumerate(arr)]
    else:
        heap = [(-val, idx) for idx, val in enumerate(arr)]

    heapq.heapify(heap)

    # Initialize a list to store the indices of the extreme values
    indices = []

    # Pop elements from the heap to get the indices of the extreme values
    for _ in range(n):
        _, idx = heapq.heappop(heap)
        indices.append(idx)

    return indices
def print_observation(obs, nr_machines):
    # o_pending = obs["pending_jobs"]
    o_machines = obs["machines"]
    o_machine_recipes = obs["machine_recipes"]
    o_p_remaining = obs["pending_job_remaining_times"]
    #o_p_steps_to_deadline = obs["pending_job_steps_to_deadline"]
    # o_ip_remaining = obs["inprogress_job_remaining_times"]
    machines_matrix = o_machines.reshape((nr_machines, -1))
    # print(TextColors.YELLOW + "Pending Jobs:\n" + TextColors.RESET, o_pending)
    # print("Recipes?:\n",o_recipes)
    print(
        TextColors.YELLOW + "Remaining time for jobs pending:\n" + TextColors.RESET,
        o_p_remaining,
    )
    # print(
    #     TextColors.YELLOW + "Steps to deadline for jobs pending:\n" + TextColors.RESET,
    #     o_p_steps_to_deadline,
    # )
    # print(TextColors.YELLOW+"Remaining time for jobs ip:\n"+TextColors.RESET, o_ip_remaining)
    print(TextColors.YELLOW + "Machine occupancy:" + TextColors.RESET)
    print(TextColors.GREEN + "       J0  J1  J2   TC" + TextColors.RESET)
    for i in range(nr_machines):
        print(TextColors.GREEN + "M", i, " " + TextColors.RESET + "[ ", end="")
        for j in range(len(machines_matrix[i]) - 1):
            if machines_matrix[i][j] == 1:
                # print(TextColors.GREEN+"M",i," "+TextColors.RESET,machines_matrix[i])
                print(TextColors.CYAN + "1.  " + TextColors.RESET, end="")
            else:
                print("0.  ", end="")
        print(f'{machines_matrix[i][-1]}  ', end="")
        print("]")
    print()

def print_capacity_obs(obs, env):
    # o_pending = obs["pending_jobs"]
    obs_jobs_queue = obs["new_jobs_queue"]
    obs_jobs_queue_capacities = obs["new_jobs_tray_capacities"]
    obs_pending_machine_capacity = obs["machine_pending_capacity"]
    obs_active_machine_capacity = obs["machine_active_capacity"]
    obs_active_machine_recipe = obs["machine_active_recipe"]
    obs_machine_recipes = obs["machine_recipes"]
    #obs_pending_job_remaining_time = obs["pending_job_remaining_times"]
    obs_pending_job_steps_to_deadline = obs["pending_job_steps_to_deadline"]

    obs_pending_job_recipes = obs["pending_job_recipe"]
    obs_p_job_process_time_deadline_ratio = obs["pending_job_process_time_deadline_ratio"]
    obs_p_job_tray_capacities = obs["pending_job_tray_capacities"]
    # obs_p_job_next_recipes: np.ndarray = obs['pending_job_next_recipes']
    # obs_uc_job_next_recipes: np.ndarray = obs['uncompleted_job_buffer_next_recipes']
    # obs_uc_job_process_time_deadline_ratio = obs["uncompleted_job_buffer_process_time_deadline_ratio"]
    # # obs_uc_job_recipes = obs["uncompleted_job_recipes"]
    # obs_uc_job_buffer_recipes = obs["uncompleted_job_buffer_recipes"]
    # #obs_uc_job_remaining_times = obs["uncompleted_job_remaining_times"]
    # obs_uc_job_buffer_remaining_times = obs["uncompleted_job_remaining_times"]
    # obs_pending_job_recipe_count = obs["pending_job_recipe_count"]
    # obs_lost_jobs_count = obs["lost_jobs_count"]
    # # obs_uc_job_recipe_count = obs["uncompleted_job_recipe_count"]
    # obs_uc_job_buffer_recipe_count = obs["uncompleted_job_buffer_recipe_count"]

    # o_p_steps_to_deadline = obs["pending_job_steps_to_deadline"]
    # o_ip_remaining = obs["inprogress_job_remaining_times"]


    # print(TextColors.YELLOW + "Pending Jobs:\n" + TextColors.RESET, o_pending)
    # print("Recipes?:\n",o_recipes)
    # print(
    #     TextColors.YELLOW + "Remaining time for jobs pending:\n" + TextColors.RESET,
    #     obs_pending_job_remaining_time,
    # )
    # print(
    #     TextColors.YELLOW + "Steps to deadline for jobs pending:\n" + TextColors.RESET,
    #     o_p_steps_to_deadline,
    # )
    # print(TextColors.YELLOW+"Remaining time for jobs ip:\n"+TextColors.RESET, o_ip_remaining)
    print(
        TextColors.YELLOW + "Jobs queue:\n" + TextColors.RESET,
        obs_jobs_queue,
    )
    print(
        TextColors.YELLOW + "Jobs queue capacities:\n" + TextColors.RESET,
        obs_jobs_queue_capacities,
    )
    print(
        TextColors.YELLOW + "Pending job steps to deadline:\n" + TextColors.RESET,
        obs_pending_job_steps_to_deadline,
    )
    print(
        TextColors.YELLOW + "Pending jobs process time to deadline ratio:\n" + TextColors.RESET,
        obs_p_job_process_time_deadline_ratio,
    )
    print(TextColors.YELLOW + "Pending job recipes:" + TextColors.RESET, obs_pending_job_recipes)
    print(TextColors.YELLOW + "Pending job tray capacities:" + TextColors.RESET, obs_p_job_tray_capacities)
    # print(TextColors.YELLOW + "Uncompleted job buffer recipes:" + TextColors.RESET, obs_uc_job_buffer_recipes)
    # print(TextColors.YELLOW + "UC job buffer process time to deadline:" + TextColors.RESET, obs_uc_job_process_time_deadline_ratio)
    # print(TextColors.YELLOW + "Uncompleted job  buffer remaining times:" + TextColors.RESET, obs_uc_job_buffer_remaining_times)
    # print(TextColors.YELLOW + "Pending job recipe count:" + TextColors.RESET, obs_pending_job_recipe_count)
    # print(TextColors.YELLOW + "Uncompleted job buffer recipe count:" + TextColors.RESET, obs_uc_job_buffer_recipe_count)
    print(TextColors.YELLOW + "Machine active capacity utilization:" + TextColors.RESET, obs_active_machine_capacity)
    print(TextColors.YELLOW + "Machine pending capacity utilization:" + TextColors.RESET, obs_pending_machine_capacity)
    print(TextColors.YELLOW + "Machine active recipes:" + TextColors.RESET, obs_active_machine_recipe)
    print(TextColors.YELLOW + "Machine recipes:" + TextColors.RESET,
          obs_machine_recipes.reshape(len(env.get_machines()), env.get_available_recipes_count()))

    # print(TextColors.YELLOW + "Pending job next recipes:" + TextColors.RESET,
    #       obs_p_job_next_recipes.reshape(env.get_buffer_size(), env.get_max_next_recipes()))
    # print(TextColors.YELLOW + "Pending job next recipes:" + TextColors.RESET,
    #       obs_uc_job_next_recipes.reshape(env.get_buffer_size(), env.get_max_next_recipes()))
    # print(TextColors.YELLOW + "Lost jobs count:" + TextColors.RESET, obs_lost_jobs_count)

    # for i in range(n_machines):
    #     print(TextColors.GREEN + "M", i, " " + TextColors.RESET + "[ ", end="")
    #     for j in range(len(machines_capacity_matrix[i])):
    #         if machines_matrix[i][j] == 1:
    #             # print(TextColors.GREEN+"M",i," "+TextColors.RESET,machines_matrix[i])
    #             print(TextColors.CYAN + "1.  " + TextColors.RESET, end="")
    #         else:
    #             print("0.  ", end="")
    #     print(f'{machines_matrix[i][-1]}  ', end="")
    #     print("]")
    # print()
def print_scheduled_jobs(env, print_length=10, buffer_size=3):
    # o_pending = obs["pending_jobs"]
    o_machines = env.get_machines()
    nr_machines = len(o_machines)
    # o_p_steps_to_deadline = obs["pending_job_steps_to_deadline"]
    # o_ip_remaining = obs["inprogress_job_remaining_times"]
    machines_matrix = env.get_machine_scheduled_jobs_matrix()
    machines_capacity_matrix = [['.' for i in range(print_length)] for m in range(nr_machines)]
    # Print active machine capacity
    for idx, m in enumerate(o_machines):
        print(TextColors.GREEN + "M", idx, " " + TextColors.RESET + "[ ", end="")
        last_job_cap = 0
        for i, job in enumerate(m.get_active_jobs()):
            j_cap = int(job.get_tray_capacity() / m.get_tray_capacity() * print_length)

            machines_capacity_matrix[idx][last_job_cap: j_cap] = [(
                                                                      TextColors.BLUE) + "#" if i % 2 == 0 else TextColors.RED + "#"] * j_cap

            last_job_cap = j_cap

        for j in range(len(machines_capacity_matrix[idx])):
            print(machines_capacity_matrix[idx][j] + " " + TextColors.RESET, end="")

        print("]")
    print()

    # Generate the string for the column headers dynamically
    job_indices = ''.join([f"J{i}  " for i in range(buffer_size)])

    print(TextColors.YELLOW + "Machine scheduled jobs:" + TextColors.RESET)
    print(TextColors.GREEN + "       " + job_indices + TextColors.RESET)
    for i in range(nr_machines):
        print(TextColors.GREEN + "M", i, " " + TextColors.RESET + "[ ", end="")
        for j in range(len(machines_matrix[i])):
            if machines_matrix[i][j] >= 1:
                # print(TextColors.GREEN+"M",i," "+TextColors.RESET,machines_matrix[i])
                print(TextColors.CYAN + f"{machines_matrix[i][j]:.0f}.  " + TextColors.RESET, end="")
            else:
                print("0.  ", end="")
        print("]")
    print()

def print_jobs(env):
    print("#### Jobs ####")
    for j in env.get_pending_jobs():
        print(j)

    print("####")

def print_uncompleted_jobs(env):
    print("#### Uncompleted Jobs ####")
    for j in env.get_uncompleted_jobs():
        print(j)

    print("####")

def print_uncompleted_jobs_buffer(env):
    print("#### Uncompleted Jobs Buffer ####")
    for j in env.get_uncompleted_jobs_buffer():
        print(j)

    print("####")

def min_max_norm(x:float, x_min: float, x_max: float):
    """
    @param x Value to normalize
    @param x_min Minimum value in dataset
    @param x_max Maximum value in dataset
    """
    #print(f"x: {x}, x_min: {x_min}, x_max: {x_max}")
    if x_min == x_max:
        return 0

    return (x - x_min)/(x_max - x_min)

def min_max_norm_list(arr: list[float]):
    x_min = min(arr)
    x_max = max(arr)

    new_arr = np.array([min_max_norm(x, x_min, x_max) for x in arr])

    return new_arr


def create_bins(input_array, group_size=10):
    # Reshape the array into a 2D array with the specified group size
    reshaped_array = np.reshape(input_array, (len(input_array) // group_size, group_size))

    # Calculate the mean along the second axis (axis=1)
    mean_array = np.mean(reshaped_array, axis=1)

    return mean_array


def save_agent_results(rewards, tardiness, jot, jnot, path:str = "files/data/"):
    data = {
        "rewards": rewards,
        "tardiness": tardiness,
        "jot": jot,
        "jnot": jnot}
    with open(path, "wb") as file:
        pickle.dump(data, file)


def load_agent_results(path:str = "files/data/agent_data_1000.pkl"):
    with open(path, "rb") as file:
        data = pickle.load(file)
    return data["rewards"], data["tardiness"], data["jot"], data["jnot"]