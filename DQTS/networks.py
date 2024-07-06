import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os

import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from parameter import get_args

args = get_args()

# Hyper Parameters
BATCH_SIZE = 32             # 
LR = 0.01                   # learning rate
EPSILON = 0.9               # greedy policy
GAMMA = 0.9                 # reward discount
MEMORY_CAPACITY = 2000      # replay buffer size
TARGET_REPLACE_ITER = 100   # target update frequency Q-target网络的更新频率
N_ACTIONS = args.VM_num
N_STATES = 1 + args.VM_num # N_STATES = env.observation_space.shape[0]

# 定义神经网络
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 50)     # 定义神经网络线性层, 输入神经元个数为state的个数, 输出神经元个数为10(隐藏层（hidden layer）设为10个神经元)
        self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.out = nn.Linear(50, N_ACTIONS)    # 输入神经元个数为10, 输出神经元个数为action的个数
        self.out.weight.data.normal_(0, 0.1)   # initialization

        # Dueling DQN
        # 优势函数
        self.advantage = nn.Sequential(
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, N_ACTIONS) 
        )

        # 价值函数
        self.value = nn.Sequential(
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, 1)
        )
        

    # 调用 forward 方法，返回前向传播的结果
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)

        
        # Dueling DQN
        '''
        x = x.view(x.size(0), -1)
        advantage = self.advantage(x)
        value = self.value(x)
        return value + advantage - advantage.mean()
        '''
        return actions_value