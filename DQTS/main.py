import os
import seaborn as sns
import matplotlib.pyplot as plt
import torch as T
import os
from dqts import DQTS
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from workflow import *
from functions import *
from env import *
from parameter import get_args

if __name__ == '__main__':
    scientific_workflow = Scientific_Workflow('Sipht', 968)
    dag = scientific_workflow.get_workflow()
    args = get_args()
    env = MultiAgentEnv(dag)

    # DQN
    dqn = DQTS()

    steps_epoch = 1000
    dqn_start_learn = 2000
    global_step = 0  #总体步数
   
    makespan_list = []
    cost_list = []
    reward_list = []
    success_rate_list = []

    result_dir = os.path.dirname(os.path.realpath(__file__)) + '\SavedResult'   
    
    for episode in range(steps_epoch + 1):
        print('----------------------------Episode', episode, '----------------------------')
        env.reset()  # 重置环境
        done_reset = False # 结束条件为工作流中的所有任务都完成
        done_goal = env.task_events # 记录任务完成状态

        step = 0
        return_ = [] # record reward for each episode

        while not done_reset:
            #print('----------------------Step', step, '----------------------')
            global_step += 1
            env.task_load()
            
            task = env.task_pop()
            task_attrs = [row for row in dag.jobs if row['id'] == task] # current task info
            wait_times = env.vm_events[2, :] # vm_waiting_time
            DQN_state = np.hstack(([task_attrs[0]['privacy_security_level']], wait_times))
            if global_step != 1:
                dqn.store_transition(last_state, last_action, last_reward, DQN_state)
            action = dqn.choose_action(DQN_state) # choose action: vm
            env.task_allocate(task, action)
            execution_time, transfer_time = env.task_finish(action)
            reward = env.feedback(task, action)
            return_.append(reward)
                
            if global_step > dqn_start_learn:
                dqn.learn()
                
            if all(done_goal):
                done_reset = True
                reward_list.append((episode, sum(return_)))
                    
            last_state = DQN_state
            last_action = action
            last_reward = reward

            step += 1

        makespan = sum(env.response_time_list)
        print('dqts_makespan:', makespan)
        if episode % 1 == 0:
            makespan_list.append(makespan)

        cost = sum(env.vm_cost_list)
        print('dqts_cost:', cost)
        if episode % 1 == 0:
            cost_list.append(cost)

        success_rate = np.sum(env.success_event) / env.task_num
        print('dqts_success_rate:', success_rate)
        if episode % 1 == 0:
            success_rate_list.append(success_rate)

    # save data
    np.save(result_dir + '/dqts_makespan.npy', makespan_list)
    np.save(result_dir + '/dqts_cost.npy', cost_list)
    np.save(result_dir + '/dqts_success_rate.npy', success_rate_list)
    np.save(result_dir + '/dqts_reward.npy', reward_list)