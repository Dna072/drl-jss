from custom_environment.environment_factory import init_custom_factory_env
from custom_environment.environment import FactoryEnv
from custom_environment.job import Job

def useDeadline(job: Job):
    return job.get_deadline_datetime()

def useArrival(job: Job):
    return job.get_arrival_datetime()

def shortest_deadline_first_rule(env: FactoryEnv):
    steps = 0
    while steps < 15:
        steps += 1
        
        for job in env.get_pending_jobs():
            print(job)
            print("----")

        deadline = env.get_pending_jobs()[0].get_deadline_datetime()
        job_index = 0

        for idx, job in enumerate(env.get_pending_jobs()):
            # go through the list of jobs and pick one with shortest deadline
            if deadline > job.get_deadline_datetime():
                deadline = job.get_deadline_datetime()
                job_index = idx
        
        #encode action for the job at index
        action = encode_job_action(env, job, job_index)
        env.step(action)
        print(f"action: {action}")
        print(f"obs: {env.get_obs()}")

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
        
        #encode action for the job at index
        action = encode_job_action(env, job, job_index)
        env.step(action)
        print(f"action: {action}")
        print(f"obs: {env.get_obs()}")

        #for idx, job in env.get_pending_jobs():
def encode_job_action(env: FactoryEnv, job: Job, job_index: int):
    # get first machine that can perform the job recipe
    machine_idx = 0
    for idx, machine in enumerate(env.get_machines()):
        valid_recipes = machine.get_job_valid_recipes(job=job)
        
        if len(valid_recipes) > 0:
            print(f"job_indx: {job_index} job recipes: {valid_recipes[0]}")
            machine_idx = idx
            break
    
    return machine_idx * env.get_buffer_size() + job_index


if __name__ == "__main__":
    custom_env = init_custom_factory_env()

    for job in env.get_pending_jobs():
        print(job)
        print("----")
    
    shortest_deadline_first_rule(env=custom_env)

