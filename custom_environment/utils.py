#from custom_environment.environment import FactoryEnv


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