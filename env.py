"""
Code for creating a multi-agent environment with VMs and tasks 

定义VM, workflow中的task放入queue进行调度
"""
import queue
from workflow import *
from parameter import get_args
import random

args = get_args()

class MultiAgentEnv:
    def __init__(self, dag):
        """
        Function for initializing the environment
        VM, dag, task, queue
        """
        self.dag = dag
        
        # VM
        self.vm_num = args.VM_num # VM number
        self.private_cloud_num = args.private_cloud_num # private cloud VM number
        self.public_cloud_num = args.public_cloud_num # public cloud VM number
        self.vm_cost = args.vm_cost # VM price
        self.private_cloud_cost = args.private_cloud_cost # private cloud VM price
        self.public_cloud_cost = args.public_cloud_cost # public cloud VM price
        self.vm_capacity = args.vm_capacity # VM computing capacity
        self.private_cloud_capacity = args.private_cloud_capacity # private cloud VM computing capacity
        self.public_cloud_capacity = args.public_cloud_capacity # public cloud VM computing capacity
        self.vm_speed = args.vm_speed # VM computing speed
        self.private_cloud_speed = args.private_cloud_speed # private cloud VM computing speed
        self.public_cloud_speed = args.public_cloud_speed # public cloud VM computing speed
        self.vm_type = args.vm_type # VM type: 0 for private cloud VM, 1 for public cloud VM
        self.vm_security = args.vm_security

        # queues for vms
        self.vm_queues = {i: queue.Queue() for i in range(self.vm_num)}

        # task
        self.task_num = dag.n_task # task number

        # edge
        self.edges = self.dag.edges
        self.edge_events = np.zeros((self.task_num, self.task_num)) # record edge transfer completion state, 1 for finished, 0 for unfinished

        # record state
        # self.task_events = np.zeros((self.task_num)) # task finish state, 1 for finished, 0 for unfinished
        self.task_events = [False] * self.task_num
        # vm state, including vm idle time and tasks on vm: [vm_idle, task_id]
        # vm state, including [task_num_of_vm, executing_task_time, wating_task_time]
        self.vm_events = np.zeros((4, self.vm_num))  # [task_num_of_vm, executing_task_time, wating_task_time, task_num_of_queue]
        self.dqn_events = np.zeros((7, self.task_num)) # dqn state, including .... 
        '''
        dqn_events:
        self.dqn_events[0, action] = excution_time
        self.dqn_events[1, action] = reward
        self.dqn_events[2, action] = task_arrival_time   # arrival at Queue
        self.dqn_events[3, action] = task_start_time
        self.dqn_events[4, action] = task_waiting_time    # task_start_time - task_arrival_time
        self.dqn_events[5, action] = task_finish_time
        self.dqn_events[6, action] = tranfer_time
        '''
        # state for agent 
        self.task_num_of_vm = np.zeros(self.vm_num) # task number on each vm
        self.executing_time = np.zeros(self.vm_num) # executing time on each vm
        self.waiting_time = np.zeros(self.vm_num) # waiting time of each vm
        self.obs = np.column_stack((self.task_num_of_vm, self.executing_time, self.waiting_time)) # state of each agent

        # record for makespan
        self.response_time_list = []

        # record for cost
        self.vm_cost_list = []

        # record for success rate
        self.success_event = np.zeros((self.task_num))

        # queue for dag
        self.dag_queue = Queue(dag)


    def reset(self):
        # edge
        self.edge_events = np.zeros((self.task_num, self.task_num))
        # queues for vms
        self.vm_queues = {i: queue.Queue() for i in range(self.vm_num)}
        # self.task_events = np.zeros((self.task_num)) 
        self.task_events = [False] * self.task_num
        self.vm_events = np.zeros((4, self.vm_num)) 
        self.dqn_events = np.zeros((7, self.task_num))
        self.dag_queue = Queue(self.dag)
        # state for agent 
        self.task_num_of_vm = np.zeros((self.vm_num)) # task number on each vm
        self.executing_time = np.zeros((self.vm_num)) # executing time on each vm
        self.waiting_time = np.zeros((self.vm_num)) # waiting time of each vm
        self.obs = np.column_stack((self.task_num_of_vm, self.executing_time, self.waiting_time)) # state of each agent
        # record for makespan
        self.response_time_list = []
        # record for cost
        self.vm_cost_list = []
        # record for success rate
        self.success_event = np.zeros((self.task_num))
        return self.obs
    

    def update_arrival_time(self, n):
        """
        Function for updating the arrival time required for each task
        n: task id
        """
        # arrival time of tasks: update in task_load, that is, when task is put into Queue
        if len(self.dag.precursor[n]) == 0:
            self.dqn_events[2, n] = 0
        else:
            max_time = 0
            for pred_n in self.dag.precursor[n]:
                if(self.dqn_events[5, pred_n] > max_time):
                    max_time = self.dqn_events[5, pred_n]
            self.dqn_events[2, n] = max_time


    def update_start_time(self, n, vm_id):
        """
        Function for updating the start time required for each task
        n: task id
        """
        # start time of tasks
        if self.vm_queues[vm_id].qsize() == 1: # allocate后，vm上只有一个task，即allocate后直接start
        # if self.vm_events[0, vm_id] == 1: # allocate后，vm上只有一个task，即allocate后直接start
            self.dqn_events[3, n] = self.dqn_events[2, n]
        else: # allocate后，vm上有多个task，即allocate后等待前面的task完成后才start
            self.dqn_events[3, n] = self.dqn_events[2, n] + self.vm_events[2, vm_id]


    def update_waiting_time(self, n):
        """
        Function for updating the waiting time required for each task
        n: task id
        """
        # waiting time of tasks: task_start_time - task_arrival_time
        self.dqn_events[4, n] = self.dqn_events[3, n] - self.dqn_events[2, n]

    
    def update_finish_time(self, n, vm_id):
        """
        Function for updating the finish time required for each task
        n: task id
        """
        # finish time of tasks: update in task_finish
        self.dqn_events[5, n] = self.dqn_events[3, n] + self.dqn_events[0, n] + self.dqn_events[6, n]


    # 总体queue的操作
    def task_load(self):
        """
        Function for loading tasks into the dag_queue
        """
        unfinished_task = [i for i in range(self.task_num) if self.task_events[i] == False]
        # print('unfinished_task:', unfinished_task)
        # 加载任务：满足所有父任务完成且传输完毕，且不在队列中
        for n in range(self.task_num):
            if(self.dag_queue.check_legal(n, self.task_events) and n not in self.dag_queue.task_queue.queue and n in unfinished_task):
                if all(self.edge_events[pred_n, n] == 1 for pred_n in self.dag.precursor[n]): # 判断所有父任务传输完毕
                    self.dag_queue.push(n)
                    self.update_arrival_time(n) # task arrival time
                    

    def task_pop(self):
        """
        Function for popping a task from the dag_queue
        """
        task = self.dag_queue.pop()
        # print('Queue:', self.dag_queue.task_queue.queue)
        return task


    # vm即agent的操作
    def task_allocate(self, action, vm_id):
        """
        Function for allocating a task to a VM
        :param action: the action of the agent, which is the task id
        """
        selected_task = self.dag.jobs[action]
        execution_time = selected_task['runtime'] / self.vm_speed[vm_id]
        # self.executing_time_list.append(execution_time) # for makespan

        if self.vm_queues[vm_id].empty() == True:
            self.vm_events[1, vm_id] = execution_time
        else:
            self.vm_events[2, vm_id] += execution_time
        self.vm_queues[vm_id].put(selected_task)
        self.vm_events[0, vm_id] += 1
        self.update_start_time(action, vm_id) # task start time
        # self.vm.events[3, vm_id] += 1

        
    def task_finish(self, vm_id):
        """
        Function for finishing a task
        """
        finished_task = self.vm_queues[vm_id].get()
        execution_time = finished_task['runtime'] / self.vm_speed[vm_id]
        task_id = finished_task['id']
        
        total_transfer_time = 0
        for succ_n in self.dag.successor[task_id]:
            transfer_time = self.edges[task_id, succ_n] / 1000000
            self.edge_events[task_id, succ_n] = 1
            total_transfer_time += transfer_time
        self.dqn_events[6, task_id] = total_transfer_time

        self.vm_events[0, vm_id] -= 1
        
        if self.vm_queues[vm_id].empty():
            self.vm_events[1, vm_id] = 0
        else:
            self.vm_events[1, vm_id] = self.vm_queues[vm_id].queue[0]['runtime'] / self.vm_speed[vm_id]
            self.vm_events[2, vm_id] -= execution_time
        self.task_events[finished_task['id']] = True
        # print('task_events:', self.task_events)
        self.dqn_events[0, task_id] = execution_time
        self.update_waiting_time(task_id) # task waiting time
        self.update_finish_time(task_id, vm_id) # task finish time
        self.task_load() # 完成一个task就进行一遍task_load，将满足条件的task放入队列
        return execution_time, total_transfer_time
    

    def feedback(self, action, vm_id):
        """
        Function for getting the feedback of environment 
        """
        execution_time = self.dqn_events[0, action]
        # print('execution_time:', execution_time)
        waiting_time = abs(self.dqn_events[4, action])
        # print('waiting_time:', waiting_time)
        transfer_time = self.dqn_events[6, action]
        # print('transfer_time:', transfer_time)
        response_time = execution_time + waiting_time + transfer_time
        self.response_time_list.append(response_time) # for makespan
        # print('response_time:', response_time)

        vm_cost = execution_time * self.vm_cost[vm_id]
        self.vm_cost_list.append(vm_cost) # for vm cost

        # rand_num = random.random()
        # security = 0 if rand_num < 0.3 else 1  

        security = 0  # 暂替，确认 scheduling model 后补全
        if self.vm_security[vm_id] == 1:
            security = 1

        suc = 0
        if self.dag.jobs[action]['privacy_security_level'] == 1:
            if self.vm_type[vm_id] == 0:
                suc = 1
        elif self.dag.jobs[action]['privacy_security_level'] == 2:
            if self.vm_type[vm_id] == 0:
                suc = 1
            else:
                if security == 1:
                    suc = 1
        elif self.dag.jobs[action]['privacy_security_level'] == 3:
            if self.vm_type[vm_id] == 0:
                suc = 1
            else:
                if security == 1:
                    suc = 1

        if suc == 1:
            self.success_event[action] = 1 # for success rate


        reward = 0
        if response_time == 0:
            reward = 0
        else:
            if suc == 0:
                reward = -10
            else:
                r1 = np.exp(-(1 * response_time)) / 0.1
                # print('r1:', r1)
                r2 = np.exp(-(1 * vm_cost)) / 0.1
                # print('r2:', r2)
                reward = r1 + r2
        # print('reward:', reward)
        self.dqn_events[1, action] = reward
        return reward

    def get_state(self):
        vm_tasknum = self.get_VM_tasknum()
        vm_executing = self.get_VM_executing()
        vm_waiting = self.get_VM_waiting()
        queue_size = self.dag_queue.task_queue.qsize() 
        queue_length = np.full((self.vm_num), queue_size)
        obs = np.column_stack((vm_tasknum, vm_executing, vm_waiting, queue_length))
        return obs

    def get_VM_tasknum(self):
        vm_tasknum = self.vm_events[0, :]
        return vm_tasknum

    def get_VM_executing(self):
        vm_executing = self.vm_events[1, :]
        return vm_executing

    def get_VM_waiting(self):
        vm_waiting = self.vm_events[2, :]
        return vm_waiting




class Queue:
    def __init__(self, dag):
        """
        Function for initializing the task queue
        """
        self.dag = dag
        self.task_queue = queue.Queue(maxsize=dag.n_task)
        self.task_count = 0
        

    def push(self, task):
        """
        Function for pushing a task into the queue
        :param task: the task to be pushed
        :return: the queue after pushing
        """
        self.task_queue.put(task)
        return self.task_queue


    def pop(self):
        """
        Function for popping a task from the queue
        :return: the task popped
        """
        task = self.task_queue.get()
        return task


    def is_empty(self):
        """
        Function for checking whether the queue is empty
        :return: True if the queue is empty, False otherwise
        """
        return self.task_queue.empty()

    
    def size(self):
        """
        Function for checking the size of the queue
        :return: the size of the queue
        """
        return self.task_queue.qsize()


    def check_legal(self, n, task_finish_events):
        """
        Function for checking whether the task put in queue is legal, that is, all the precursor tasks are finished
        :return: True if the queue is legal, False otherwise
        """
        if len(self.dag.precursor[n]) == 0:
            return True
        for pred_n in self.dag.precursor[n]:
            if(task_finish_events[pred_n] == False):
                return False    
        return True


    def check_succ(self, n, finish_vertexs, wait_vertexs):
        """
        Function for 
        返回一个 "n个后续任务合法且不在wait_vertexs中 "的任务的列表
        """
        list = []
        for succ_n in self.dag.successor[n+1]:
            if(self.check_legal(succ_n, finish_vertexs) and succ_n not in wait_vertexs):
                list.append(succ_n)
        return list
