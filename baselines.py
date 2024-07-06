import numpy as np
from env import *
from parameter import get_args

args = get_args()

class Baselines:
    def __init__(self, dag):
        self.dag = dag
        self.env = MultiAgentEnv(dag)

    # random
    def random(self):
        self.env.reset()
        makespan_random = 0
        vm_cost_random = 0
        done_goal = self.env.task_events
        while not all(done_goal):
            self.env.task_load()
            for i in range(self.env.dag_queue.size()):
                task = self.env.task_pop()
                vm_selected = np.random.randint(self.env.vm_num)
                self.env.task_allocate(task, vm_selected)
                execution_time, transfer_time = self.env.task_finish(vm_selected)            
                self.env.feedback(task, vm_selected)
        makespan_random = sum(self.env.response_time_list)
        vm_cost_random = sum(self.env.vm_cost_list)
        success_random = np.sum(self.env.success_event) / self.env.task_num
        return makespan_random, vm_cost_random, success_random
    
    # earliest
    def earliest(self):
        self.env.reset()
        makespan_earliest = 0
        vm_cost_earliest = 0
        done_goal = self.env.task_events
        while not all(done_goal):
            self.env.task_load()
            for i in range(self.env.dag_queue.size()):
                task = self.env.task_pop()
                vm_selected = np.argmin(self.env.vm_events[2, :]) # 4: vm_waiting_time
                self.env.task_allocate(task, vm_selected)
                execution_time, transfer_time = self.env.task_finish(vm_selected)
                self.env.feedback(task, vm_selected)
        makespan_earliest = sum(self.env.response_time_list)
        vm_cost_earliest = sum(self.env.vm_cost_list)
        success_earliest = np.sum(self.env.success_event) / self.env.task_num
        return makespan_earliest, vm_cost_earliest, success_earliest
    
    # round robin
    def round_robin(self):
        self.env.reset()
        makespan_round_robin = 0
        vm_cost_round_robin = 0
        done_goal = self.env.task_events
        task_count = 0
        while not all(done_goal):
            self.env.task_load()
            for i in range(self.env.dag_queue.size()):
                task = self.env.task_pop()
                vm_selected = task_count % self.env.vm_num
                self.env.task_allocate(task, vm_selected)
                execution_time, transfer_time = self.env.task_finish(vm_selected)
                self.env.feedback(task, vm_selected)
        makespan_round_robin = sum(self.env.response_time_list)
        vm_cost_round_robin = sum(self.env.vm_cost_list)
        success_round_robin = np.sum(self.env.success_event) / self.env.task_num
        return makespan_round_robin, vm_cost_round_robin, success_round_robin
