from custom_environment.environment_factory import init_custom_factory_env
from custom_environment.environment import FactoryEnv
from custom_environment.job import Job
from custom_environment.my_factory_env import MyFactoryEnv
import matplotlib.pyplot as plt

def useDeadline(job: Job):
    return job.get_deadline_datetime()

def useArrival(job: Job):
    return job.get_arrival_datetime()

def shortest_deadline_first_rule(env: MyFactoryEnv):
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
        if(len(env.get_pending_jobs()) == 0):
            break

        # set inital deadline and job index
        deadline = env.get_pending_jobs()[0].get_deadline_datetime()
        job_index = 0
        job_todo = env.get_pending_jobs()[0]

        for job in env.get_pending_jobs():
            # go through the list of jobs and pick one with shortest deadline
            if deadline > job.get_deadline_datetime():
                deadline = job.get_deadline_datetime()
                job_todo = job
        
        #encode action for the job at index
        action = encode_job_action(env, job_todo)
        
        # check if there are any available machines before stepping
        if env.is_machine_available(action):
            #print("Machines available, stepping")
            #print(f"Obs: {env.get_obs()}")
            obs, reward, terminated, truncated, info = env.step(action)

            rewards.append(reward)
            avg_machine_utilization.append(env.get_average_machine_utilization_time_percentage())
            avg_machine_idle_time.append(env.get_average_machine_idle_time_percentage())

            print(f"action: {action}")
            print(f"obs: {env.get_obs(flatten=False)}")
        else:
            env._update_factory_env_state()

    # plot rewards
    fig, axs = plt.subplots(2,2)
    fig.suptitle('Shortest deadline first rule')
    axs[0,0].plot(rewards)
    axs[0,0].set_title('Rewards plot')
    axs[0,0].set(ylabel='Reward')

    axs[1,1].plot(avg_machine_utilization)
    axs[1,1].set_title('Avg Machine utilization')
    axs[1,1].set(ylabel='Avg Utilization %')

    axs[1,0].plot(avg_machine_idle_time)
    axs[1,0].set_title('Avg Machine idle time')
    axs[1,0].set(ylabel='Avg idleness %')

    for ax in axs.flat:
        ax.set(xlabel='Time step')

    for ax in axs.flat:
        ax.label_outer()
    #plt.plot(rewards)
    plt.show()
    plt.savefig('files/plots/shortest_deadline_plot1.png', format='png')
       

def first_in_first_out_rule(env: FactoryEnv):
    steps = 0
    while steps < 15:
        steps += 1

        arrival = env.get_pending_jobs()[0].get_arrival_datetime()
        job_index = 0

        for idx, job in enumerate(env.get_pending_jobs()):
            # go through the list of jobs and pick one with shortest deadline
            if arrival < job.get_arrival_datetime():
                arrival = job.get_deadline_datetime()
                job_index = idx

                print(f"JobIndx: {job_index}, Job: {job}")
        
        #encode action for the job at index
        action = encode_job_action(env, job, job_index)
        env.step(action)
        print(f"action: {action}")
        print(f"obs: {env.get_obs()}")

        #for idx, job in env.get_pending_jobs():

def encode_job_action(env: FactoryEnv, job: Job):
    # get first machine that can perform the job recipe
    #print(f"Encoding job: {job.get_id()}")
    machine_idx = 0
    for idx, machine in enumerate(env.get_machines()):
        valid_recipes = machine.get_job_valid_recipes(job=job)
        
        if len(valid_recipes) > 0:
            #print(f"MachineID: {machine.get_factory_id()} Recipes: {len(valid_recipes)}")
            #print(f"job_indx: {job.get_id()} job_id: {job.get_factory_id()} job recipes: {valid_recipes[0]}")
            machine_idx = idx
            break
    
    return machine_idx * env.get_buffer_size() + job.get_id()


if __name__ == "__main__":
    custom_env = init_custom_factory_env()

    for job in custom_env.get_pending_jobs():
        print(job)
        print("----")
    
    shortest_deadline_first_rule(env=custom_env)

