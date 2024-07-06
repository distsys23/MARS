import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from maddpg import MADDPG
from buffer import MultiAgentReplayBuffer

import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from workflow import *
from functions import *
from env import *
from parameter import get_args

if __name__ == '__main__':
    scientific_workflow = Scientific_Workflow('CyberShake', 30)
    dag = scientific_workflow.get_workflow()
    args = get_args()
    env = MultiAgentEnv(dag)
    n_agents = env.vm_num
    obs_space = 4  
    actor_dims = []
    for idx in range(n_agents):
        actor_dims.append(obs_space)
    critic_dims = sum(actor_dims)

    n_actions = env.task_num
    chkpt_dir = os.path.dirname(os.path.realpath(__file__)) + '\Saved'
    maddpg_agents = MADDPG(actor_dims, critic_dims, n_agents, n_actions, 
                           fc1=64, fc2=64,  
                           alpha=0.01, beta=0.01, scenario=env,
                           chkpt_dir=chkpt_dir)
    memory = MultiAgentReplayBuffer(1000000, critic_dims, actor_dims, 
                    n_actions, n_agents, batch_size=1024)
    
    steps_epoch = 10000  # number of maximum episodes
    global_step = 0

    result_dir = os.path.dirname(os.path.realpath(__file__)) + '\SavedResult'

    makespan_list = []
    cost_list = []
    reward_list = []
    success_rate_list = []

    for episode in range(steps_epoch + 1):
        print('----------------------------Episode', episode, '----------------------------')
        n_obs = env.reset()
        done_reset = False 
        done_goal = env.task_events 

        step = 0
        reward = np.zeros(n_agents) # reward for each episode

        while not done_reset:
            print('----------------------Step', step, '----------------------')
            env.task_load()  
            n_obs = env.get_state() 
            # print('n_obs:', n_obs)       
            allowed_actions = env.dag_queue.task_queue.queue  
            actions = maddpg_agents.choose_action(n_obs, allowed_actions) 

            # data normalization
            vm_speed = args.vm_speed
            actions = np.array(actions)
            vm_speed = np.array(vm_speed)
            unique_actions = np.unique(actions)

            for action in unique_actions:
                indices = np.where(actions == action)[0]
                max_index = indices[np.argmax(vm_speed[indices])]
                mask = np.ones_like(actions, dtype=bool)
                mask[indices] = False
                mask[max_index] = True
                actions[~mask] = -1

            # step: task allocate
            need_allocate = env.dag_queue.size()
            # print('need_allocate:', need_allocate)
            allocate_count = 0
            # print('Queue:', env.dag_queue.task_queue.queue)
            print('... task allocate ...')
            for j in range(env.vm_num):
                if actions[j] != -1:
                    env.task_pop()
                    env.task_allocate(actions[j], j)
                    allocate_count += 1
                    need_allocate -= 1         

            print('... task finish ...')
            for j in range(env.vm_num):
                if actions[j] != -1:
                    # print('before:', env.vm_queues[j].queue)
                    excution_time = env.task_finish(j)
                    # print('after:', env.vm_queues[j].queue)
                    reward_feedback = env.feedback(actions[j], j)
                    # print('reward_feedback:', reward_feedback)
                    reward[j] = reward_feedback
            n_obs_ = env.get_state() # state_
            # print('n_obs_:', n_obs_)
            
            # when done
            if all(done_goal):  
                done_reset = True
                reward_list.append((episode, sum(reward)))
                # print('reward_list:', reward_list)

            n_state = n_obs.reshape(1, -1)
            n_state_ = n_obs_.reshape(1, -1)
            
            print('... store transition ...')
            memory.store_transition(n_obs, n_state, actions, reward, n_obs_, n_state_, done= [False] * n_agents)

            if not memory.ready():
                pass
            else:
                # See losses in writer, which can be shown in web page
                # Type in terminal to see writer: tensorboard --logdir=SavedLoss
                print('... learning ...')
                maddpg_agents.learn(memory, global_step)

            n_obs = n_obs_
            step += 1
            global_step += 1

        makespan = sum(env.response_time_list)
        print('makespan:', makespan)
        if episode % 1 == 0:
            makespan_list.append(makespan)

        cost = sum(env.vm_cost_list)
        print('cost:', cost)
        if episode % 1 == 0:
            cost_list.append(cost)

        success_rate = np.sum(env.success_event) / env.task_num
        print('success_rate:', success_rate)
        if episode % 1 == 0:
            success_rate_list.append(success_rate)

    '''
    # draw makespan
    fig = plt.figure(figsize=(10,7))
    xdata = np.arange(len(makespan_list))
    color = '#a6b6eb'
    sns.set(style="darkgrid", font_scale=1.5)  
    sns.lineplot(data=makespan, color=color)
    plt.ylabel("Makespan", fontsize=18)
    plt.xlabel("Episodes", fontsize=18)
    plt.show()

    # draw reward 
    y = reward_list[:, 1]
    fig = plt.figure(figsize=(10,7))
    xdata = np.arange(len(reward_list))
    color = '#a6b6eb'
    sns.set(style="darkgrid", font_scale=1.5)  
    sns.lineplot(data=y, color=color)
    plt.ylabel("Reward", fontsize=18)
    plt.xlabel("Episodes", fontsize=18)
    plt.show()
    '''

    # save data
    np.save(result_dir + '/maddpg_makespan.npy', makespan_list)
    np.save(result_dir + '/maddpg_cost.npy', cost_list)
    np.save(result_dir + '/maddpg_success_rate.npy', success_rate_list)
    np.save(result_dir + '/maddpg_reward.npy', reward_list)