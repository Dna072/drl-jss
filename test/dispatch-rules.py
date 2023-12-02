from custom_environment.dispatch_rules.environment_factory_dispatch_rules import (
    init_custom_factory_env,
)
from custom_environment.dispatch_rules.environment_wrapper_dispatch_rules import (
    EnvWrapperDispatchRules,
)
from custom_environment.dispatch_rules.job_dispatch_rules import Job
import matplotlib.pyplot as plt
from time import sleep

PAUSE_TIME_IN_SECONDS: int = 5


def use_deadline(job: Job):
    return job.get_deadline_datetime()


def use_arrival(job: Job):
    return job.get_arrival_datetime()


def shortest_deadline_first_rule(env: EnvWrapperDispatchRules):
    steps = 0
    terminated = False
    rewards = []
    avg_machine_utilization = []
    avg_machine_idle_time = []

    while not terminated:
        steps += 1

        # for job in env.get_pending_jobs():
        #     print(job)
        #     print("----")
        if len(env.get_pending_jobs()) == 0:
            break

        # set initial deadline and job index
        deadline = env.get_pending_jobs()[0].get_deadline_datetime()
        # job_index = 0
        job_todo = env.get_pending_jobs()[0]

        for job in env.get_pending_jobs():
            # go through the list of jobs and pick one with shortest deadline
            if deadline > job.get_deadline_datetime():
                deadline = job.get_deadline_datetime()
                job_todo = job

        # encode action for the job at index
        action = encode_job_action(env, job_todo)
        is_terminated = env._update_factory_env_state()

        # check if there are any available machines before stepping
        if env.is_machine_available(action):
            # print("Machines available, stepping")
            # print(f"Obs: {env.get_obs()}")
            obs, reward, terminated, truncated, info = env.step(action, is_terminated)

            rewards.append(reward)
            avg_machine_utilization.append(
                env.get_average_machine_utilization_time_percentage()
            )
            avg_machine_idle_time.append(env.get_average_machine_idle_time_percentage())

            print(f"Pause for {PAUSE_TIME_IN_SECONDS} seconds")
            sleep(PAUSE_TIME_IN_SECONDS)  # pause to capture tardiness and efficiency

            print(f"action: {action}")
            print(f"obs: {env.get_obs(is_flatten=False)}")

    # plot rewards
    fig, axs = plt.subplots(3, 1)
    fig.suptitle("Shortest deadline first rule")
    axs[0].plot(rewards)
    axs[0].set_title("Rewards plot")
    axs[0].set(ylabel="Reward")

    axs[1].plot(avg_machine_utilization)
    axs[1].set_title("Avg Machine utilization")
    axs[1].set(ylabel="Avg Utilization %")

    axs[2].plot(avg_machine_idle_time)
    axs[2].set_title("Avg Machine idle time")
    axs[2].set(ylabel="Avg idleness %")

    for ax in axs.flat:
        ax.set(xlabel="Time step")

    for ax in axs.flat:
        ax.label_outer()
    # plt.plot(rewards)
    plt.show()
    plt.savefig('../files/plots/shortest_deadline_plot_100_steps.png', format='png')


def first_in_first_out_rule(env: EnvWrapperDispatchRules):
    steps = 0
    terminated = False
    rewards = []
    avg_machine_utilization = []
    avg_machine_idle_time = []
    
    while not terminated:
        steps += 1

        if len(env.get_pending_jobs()) == 0:
            break

        arrival = env.get_pending_jobs()[0].get_arrival_datetime()
        job_todo = env.get_pending_jobs()[0]

        for job in env.get_pending_jobs():
            # go through the list of jobs and pick one with shortest deadline
            if arrival < job.get_arrival_datetime():
                arrival = job.get_deadline_datetime()
                job_todo = job

                

        # encode action for the job at index
        action = encode_job_action(env, job_todo)
        is_terminated = env._update_factory_env_state()

        if env.is_machine_available(action):
            obs, reward, terminated, truncated, info = env.step(action, is_terminated)

            rewards.append(reward)
            avg_machine_utilization.append(
                env.get_average_machine_idle_time_percentage()
            )
            avg_machine_idle_time.append(env.get_average_machine_idle_time_percentage())

            print(f"Pause for {PAUSE_TIME_IN_SECONDS} seconds")
            sleep(PAUSE_TIME_IN_SECONDS)

            print(f"action: {action}")
            print(f"obs: {env.get_obs(is_flatten=False)}")

    # plot rewards
    fig, axs = plt.subplots(3, 1)
    fig.suptitle("First-In-First-Out rule")
    axs[0].plot(rewards)
    axs[0].set_title("Rewards plot")
    axs[0].set(ylabel="Reward")

    axs[1].plot(avg_machine_utilization)
    axs[1].set_title("Avg Machine utilization")
    axs[1].set(ylabel="Avg Utilization %")

    axs[2].plot(avg_machine_idle_time)
    axs[2].set_title("Avg Machine idle time")
    axs[2].set(ylabel="Avg idleness %")

    for ax in axs.flat:
        ax.set(xlabel="Time step")

    for ax in axs.flat:
        ax.label_outer()
    # plt.plot(rewards)
    plt.show()
    plt.savefig('../files/plots/fifo_plot_100_steps.png', format='png')


        # for idx, job in env.get_pending_jobs():


def encode_job_action(env: EnvWrapperDispatchRules, job: Job):
    # get first machine that can perform the job recipe
    # print(f"Encoding job: {job.get_id()}")
    machine_idx = 0
    for idx, machine in enumerate(env.get_machines()):
        valid_recipes = machine.get_job_valid_recipes(job=job)

        if len(valid_recipes) > 0:
            # print(f"MachineID: {machine.get_factory_id()} Recipes: {len(valid_recipes)}")
            # print(f"job_indx: {job.get_id()} job_id: {job.get_factory_id()} job recipes: {valid_recipes[0]}")
            machine_idx = idx
            break

    return machine_idx * env.get_buffer_size() + job.get_id()


if __name__ == "__main__":
    custom_env: EnvWrapperDispatchRules = init_custom_factory_env(max_steps=100)

    for job in custom_env.get_pending_jobs():
        print(job)
        print("----")

    shortest_deadline_first_rule(env=custom_env)
