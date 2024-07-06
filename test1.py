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
    scientific_workflow = Scientific_Workflow('Sipht', 968)
    dag = scientific_workflow.get_workflow()
    args = get_args()
    env = MultiAgentEnv(dag)

    # baselines: random, earliest, round_robin
    baselines = Baselines(dag)
    makespan_random, cost_random, success_rate_random = baselines.random()
    makespan_earliest, cost_earliest, success_rate_earliest= baselines.earliest()
    makespan_round_robin, cost_round_robin, success_rate_round_robin = baselines.round_robin()
    print('makespan_random:', makespan_random)
    print('cost_random:', cost_random)
    print('success_rate_random:', success_rate_random)
    print('makespan_earliest:', makespan_earliest)
    print('cost_earliest:', cost_earliest)
    print('success_rate_earliest:', success_rate_earliest)
    print('makespan_round_robin:', makespan_round_robin)
    print('cost_round_robin:', cost_round_robin)
    print('success_rate_round_robin:', success_rate_round_robin)