#from custom_environment.environment import FactoryEnv
import numpy as np
import pickle

class TextColors:
    RESET = "\033[0m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"


def print_observation(obs, nr_machines):
    # o_pending = obs["pending_jobs"]
    o_machines = obs["machines"]
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

def print_capacity_obs(obs, n_machines, machines, print_length):
    # o_pending = obs["pending_jobs"]
    o_machines = obs["machine_capacity"]
    o_p_remaining = obs["pending_job_remaining_times"]
    # o_p_steps_to_deadline = obs["pending_job_steps_to_deadline"]
    # o_ip_remaining = obs["inprogress_job_remaining_times"]
    machines_capacity_matrix = [['.' for i in range(print_length)] for m in range(n_machines)]

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
    print(TextColors.YELLOW + "Machine capacity utilization:" + TextColors.RESET, o_machines)

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

    for idx, m in enumerate(machines):
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
def print_scheduled_jobs(env):
    # o_pending = obs["pending_jobs"]
    o_machines = env.get_machines()
    nr_machines = len(o_machines)
    # o_p_steps_to_deadline = obs["pending_job_steps_to_deadline"]
    # o_ip_remaining = obs["inprogress_job_remaining_times"]
    machines_matrix = env.get_machine_scheduled_jobs_matrix()
    # print(TextColors.YELLOW + "Pending Jobs:\n" + TextColors.RESET, o_pending)
    # print("Recipes?:\n",o_recipes)

    # print(
    #     TextColors.YELLOW + "Steps to deadline for jobs pending:\n" + TextColors.RESET,
    #     o_p_steps_to_deadline,
    # )
    # print(TextColors.YELLOW+"Remaining time for jobs ip:\n"+TextColors.RESET, o_ip_remaining)
    print(TextColors.YELLOW + "Machine scheduled jobs:" + TextColors.RESET)
    print(TextColors.GREEN + "       J0  J1  J2" + TextColors.RESET)
    for i in range(nr_machines):
        print(TextColors.GREEN + "M", i, " " + TextColors.RESET + "[ ", end="")
        for j in range(len(machines_matrix[i])):
            if machines_matrix[i][j] == 1:
                # print(TextColors.GREEN+"M",i," "+TextColors.RESET,machines_matrix[i])
                print(TextColors.CYAN + "1.  " + TextColors.RESET, end="")
            else:
                print("0.  ", end="")
        print("]")
    print()

def print_jobs(env):
    print("#### Jobs ####")
    for j in env.get_pending_jobs():
        print(j)

    print("####")

def min_max_norm(x:float, x_min: float, x_max: float):
    """
    @param x Value to normalize
    @param x_min Minimum value in dataset
    @param x_max Maximum value in dataset
    """
    #print(f"x: {x}, x_min: {x_min}, x_max: {x_max}")
    if x_min == 0 and x_max == 0:
        return 0

    return (x - x_min)/(x_max - x_min)


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