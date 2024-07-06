"""
Main function
Type in terminal to see writer in web page: tensorboard --logdir=SavedLoss

CUDA version: 11.2
tensorboard: 2.9.0
"""
import os
import seaborn as sns
import matplotlib.pyplot as plt
import time
import torch as T
from workflow import *
from functions import *
from buffer import MultiAgentReplayBuffer
from matd3 import MATD3
from env import *
from parameter import get_args

from torch.utils.tensorboard import SummaryWriter
from rl_plotter.logger import Logger

from baselines import Baselines

if __name__ == '__main__':
    scientific_workflow = Scientific_Workflow('Inspiral', 30)
    dag = scientific_workflow.get_workflow()
    args = get_args()
    env = MultiAgentEnv(dag)
    n_agents = env.vm_num
    # Actor的网络结构为：状态的维度--400--200--动作的维度；Critic的网络结构为状态+动作的维度--400--200--1
    obs_space = 4  # 即各agent的状态数量，[task_num_of_vm, executing_task_time, wating_task_time, task_num_of_queue]
    actor_dims = []
    for idx in range(n_agents):
        actor_dims.append(obs_space)
    critic_dims = sum(actor_dims)

    # action space, action: 0 for not allocate, 1 for id of task to allocate
    n_actions = env.task_num
    #n_actions = 2 # 0 represents not allocate, 1 represents allocate current task (top of Queue)
    chkpt_dir = os.path.dirname(os.path.realpath(__file__)) + '\SavedNetwork'
    
    learn_interval = 100
    marl_agents = MATD3(chkpt_dir, actor_dims, critic_dims, n_agents, n_actions, learn_interval,
                        fc1=32, fc2=32, alpha=0.01, beta=0.01)
    # max_size = 1000000
    # memory = MultiAgentReplayBuffer(max_size, actor_dims, critic_dims, n_agents, n_actions, batch_size=1024)
    max_size = 1000000
    memory = MultiAgentReplayBuffer(max_size, actor_dims, critic_dims, n_agents, n_actions, batch_size=1024)

    steps_epoch = 10000   #10000  # number of maximum episodes
    steps_exp = steps_epoch / 2
    global_step = 0

    result_dir = os.path.dirname(os.path.realpath(__file__)) + '\SavedResult'
    writer = SummaryWriter("SavedLoss")
    #writer_dir = os.path.dirname(os.path.realpath(__file__)) + '\SavedLoss'
    #delete_files(writer_dir)

    makespan_list = []
    cost_list = []
    reward_list = []
    success_rate_list = []

    # main loop for MATD3
    for episode in range(steps_epoch):
        print('----------------------------Episode', episode, '----------------------------')
        """
        obs: init state of agent: 
        """
        n_obs = env.reset()
        done_reset = False # 结束条件为工作流中的所有任务都完成
        done_goal = env.task_events # 记录任务完成状态
        '''[False] * self.task_num'''

        if episode < steps_exp:
            Exploration = True
            marl_agents.reset_noise()
        else:
            Exploration = False
        noise_l = 0.2  # valid noise range
        
        step = 0
        reward = np.zeros(n_agents) # reward for each episode

        while not done_reset:
            print('----------------------Step', step, '----------------------')
            env.task_load() # 加载满足条件的 task 到总队列 Queue 中  
            n_obs = env.get_state() # state before allocate
            # print('n_obs:', n_obs)       
            allowed_actions = env.dag_queue.task_queue.queue # allowed actions: tasks in queue
            # print('allowed_actions:', allowed_actions)     
            actions = marl_agents.choose_action(n_obs, allowed_actions, Exploration, noise_l) 
            # action: -1 for not allocate, others for id of task to allocate, example: actions = [-1, -1, -1, 29, -1, -1, -1, -1, -1, -1]
            
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
            # print('actions:', actions)
            '''actions: [-1 -1 -1 -1 -1 -1 13  2 -1 -1 -1 -1 -1 -1 -1 -1]'''

            # step: task allocate
            #need_allocate = sum([1 for action in actions if action in env.dag_queue.task_queue.queue])
            need_allocate = env.dag_queue.size()
            # print('need_allocate:', need_allocate)
            allocate_count = 0
            # print('Queue:', env.dag_queue.task_queue.queue)
            print('... task allocate ...')
            for j in range(env.vm_num):
                #if actions[j] in env.dag_queue.task_queue.queue:
                if actions[j] != -1:
                    env.task_pop()
                    env.task_allocate(actions[j], j)
                    # print(env.vm_queues[j].queue)
                    # print(env.dag_queue.task_queue.queue)
                    allocate_count += 1
                    need_allocate -= 1         
            # print('Queue_after:', env.dag_queue.task_queue.queue)

            # 何时判断完成任务并返回reward
            # 加入一个class vm
            # test
            # print('Queue:', env.dag_queue.task_queue.queue)
            
            # feedback: task finish and reward (def task_finish is in def feedback)
            print('... task finish ...')
            for j in range(env.vm_num):
                if actions[j] != -1:
                    # print('before:', env.vm_queues[j].queue)
                    execution_time, transfer_time = env.task_finish(j)
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
            memory.store_transition(n_obs, n_state, actions, reward, n_obs_, n_state_, done= [True] * n_agents)
 
            if not memory.ready():
                pass
            else:
                # See losses in writer, which can be shown in web page
                # Type in terminal to see writer: tensorboard --logdir=SavedLoss
                print('... learning ...')
                marl_agents.learn(memory, writer, global_step)
            

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

    writer.close()
    # save networks
    marl_agents.save_checkpoint()
    
    # save data
    np.save(result_dir + '/makespan.npy', makespan_list)
    np.save(result_dir + '/cost.npy', cost_list)
    np.save(result_dir + '/success_rate.npy', success_rate_list)
    np.save(result_dir + '/reward.npy', reward_list)
    
    '''
    # draw makespan
    fig = plt.figure(figsize=(10,7))
    xdata = np.arange(len(makespan_list))
    color = '#a6b6eb'
    sns.set(style="darkgrid", font_scale=1.5)  
    sns.lineplot(data=makespan_list, color=color)
    plt.ylabel("Makespan", fontsize=18)
    plt.xlabel("Episodes", fontsize=18)
    #plt.title("CyberShake_30", fontsize=20)
    #plt.legend(loc='lower right')
    #plt.savefig("makepan.jpg", dpi=1500)
    plt.show()

    # draw cost
    fig = plt.figure(figsize=(10,7))
    xdata = np.arange(len(cost_list))
    color = '#a6b6eb'
    sns.set(style="darkgrid", font_scale=1.5)
    sns.lineplot(data=cost_list, color=color)
    plt.ylabel("Cost", fontsize=18)
    plt.xlabel("Episodes", fontsize=18)
    plt.show()

    # draw success rate
    fig = plt.figure(figsize=(10,7))
    xdata = np.arange(len(success_rate_list))
    color = '#a6b6eb'
    sns.set(style="darkgrid", font_scale=1.5)
    sns.lineplot(data=success_rate_list, color=color)
    plt.ylabel("Success Rate", fontsize=18)
    plt.xlabel("Episodes", fontsize=18)
    plt.show()

    # draw reward 
    y = np.array(reward_list)[:, 1]
    # print('y:', y)
    fig = plt.figure(figsize=(10,7))
    xdata = np.arange(len(reward_list))
    color = '#a6b6eb'
    sns.set(style="darkgrid", font_scale=1.5)  
    sns.lineplot(data=y, color=color)
    plt.ylabel("Reward", fontsize=18)
    plt.xlabel("Episodes", fontsize=18)
    plt.show()
    '''
    



    '''
    comparison: baselines, maddpg, dqts
    '''
    # baselines: random, earliest, round_robin
    baselines = Baselines(dag)
    makespan_random, cost_random, success_rate_random = baselines.random()
    makespan_earliest, cost_earliest, success_rate_earliest= baselines.earliest()
    makespan_round_robin, cost_round_robin, success_rate_round_robin = baselines.round_robin()
    '''
    print('makespan_random:', makespan_random)
    print('cost_random:', cost_random)
    print('success_rate_random:', success_rate_random)
    print('makespan_earliest:', makespan_earliest)
    print('cost_earliest:', cost_earliest)
    print('success_rate_earliest:', success_rate_earliest)
    print('makespan_round_robin:', makespan_round_robin)
    print('cost_round_robin:', cost_round_robin)
    print('success_rate_round_robin:', success_rate_round_robin)
    '''
    
    # save baselines
    baselines_makespan = np.array([makespan_random, makespan_earliest, makespan_round_robin])
    np.save(result_dir + '/baselines_makespan.npy', baselines_makespan)
    baselines_cost = np.array([cost_random, cost_earliest, cost_round_robin])
    np.save(result_dir + '/baselines_cost.npy', baselines_cost)
    baselines_success_rate = np.array([success_rate_random, success_rate_earliest, success_rate_round_robin])
    np.save(result_dir + '/baselines_success_rate.npy', baselines_success_rate)

    
    # maddpg
    


    # dqts