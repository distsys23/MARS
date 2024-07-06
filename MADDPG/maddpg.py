import numpy as np
import torch as T
import torch.nn.functional as F
from agent import Agent

class MADDPG:
    def __init__(self, actor_dims, critic_dims, n_agents, n_actions, 
                 scenario='simple',  alpha=0.01, beta=0.01, fc1=64, 
                 fc2=64, gamma=0.99, tau=0.01, chkpt_dir='\Saved'):
        self.agents = []
        self.n_agents = n_agents
        self.n_actions = n_actions
        # chkpt_dir += scenario 
        for agent_idx in range(self.n_agents):
            self.agents.append(Agent(actor_dims[agent_idx], critic_dims,  
                            n_actions, n_agents, agent_idx, alpha=alpha, beta=beta,
                            chkpt_dir=chkpt_dir))


    def save_checkpoint(self):
        print('... saving checkpoint ...')
        for agent in self.agents:
            agent.save_models()

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        for agent in self.agents:
            agent.load_models()

    def choose_action(self, raw_obs, allowed_actions):
        actions = []
        for agent_idx, agent in enumerate(self.agents):
            action = agent.choose_action(raw_obs[agent_idx], allowed_actions)
            actions.append(action)
        return actions

    def learn(self, memory, steps_total):
        if not memory.ready():
            return

        actor_states, states, actions, rewards, \
        actor_new_states, states_, dones = memory.sample_buffer()

        device = self.agents[0].actor.device

        states = T.tensor(states, dtype=T.float).to(device)
        actions = T.tensor(np.array(actions), dtype=T.float).to(device)
        rewards = T.tensor(rewards).to(device)
        states_ = T.tensor(states_, dtype=T.float).to(device)
        dones = T.tensor(dones).to(device)
        # all these three different actions are needed to calculate the loss function
        all_agents_new_actions = []  # actions according to the target network for the new state
        all_agents_new_mu_actions = []  # actions according to the regular actor network for the current state
        old_agents_actions = []  # actions the agent actually took

        for agent_idx in range(self.n_agents):
            # actions according to the target network for the new state
            new_states = T.tensor(actor_new_states[agent_idx],
                                  dtype=T.float).to(device)
            new_pi = self.agents[agent_idx].target_actor.forward(new_states)
            all_agents_new_actions.append(new_pi)
            # actions according to the regular actor network for the current state
            mu_states = T.tensor(actor_states[agent_idx],
                                 dtype=T.float).to(device)
            pi = self.agents[agent_idx].actor.forward(mu_states)
            all_agents_new_mu_actions.append(pi)
            # actions the agent actually took
            old_agents_actions.append(actions[agent_idx])

        new_actions = all_agents_new_actions
        mu = all_agents_new_mu_actions
        old_actions = old_agents_actions

        # handle cost function
        for agent_idx in range(self.n_agents):
            # current Q estimate
            current_Q1 = self.agents[agent_idx].critic.forward(states, old_actions[agent_idx])
            # target Q value
            # with T.no_grad():
            target_Q1 = self.agents[agent_idx].target_critic.forward(states_, new_actions[agent_idx])
            #target_Q_min = T.min(target_Q1, target_Q2)
            # target_Q[dones[:, 0]] = 0.0
            target_Q = rewards[:, agent_idx] + (self.agents[agent_idx].gamma * target_Q1)
            # critic loss
            self.agents[agent_idx].critic_loss = F.mse_loss(current_Q1.float(), target_Q.float()) 

            # critic optimization
            self.agents[agent_idx].critic.optimizer.zero_grad()
            self.agents[agent_idx].critic_loss.backward()
            self.agents[agent_idx].critic.optimizer.step()


            #if steps_total % self.freq == 0 and steps_total > 0:
            if steps_total % 100 == 0 and steps_total > 0:
                # actor loss
                self.agents[agent_idx].actor_loss = -T.mean(self.agents[agent_idx].critic.Q1(states, mu[agent_idx]))
                # actor optimization
                self.agents[agent_idx].actor.optimizer.zero_grad()
                self.agents[agent_idx].actor_loss.backward()
                self.agents[agent_idx].actor.optimizer.step()
                # update parameters
                self.agents[agent_idx].update_network_parameters()





        # 11111
        '''
        actor_states, states, actions, rewards, \
        actor_new_states, states_, dones = memory.sample_buffer()

        device = self.agents[0].actor.device

        states = T.tensor(states, dtype=T.float).to(device)
        actions = T.tensor(actions, dtype=T.float).to(device)
        rewards = T.tensor(rewards).to(device)
        states_ = T.tensor(states_, dtype=T.float).to(device)
        dones = T.tensor(dones).to(device)

        all_agents_new_actions = []
        all_agents_new_mu_actions = []
        old_agents_actions = []

        for agent_idx, agent in enumerate(self.agents):
            new_states = T.tensor(actor_new_states[agent_idx], 
                                 dtype=T.float).to(device)

            new_pi = agent.target_actor.forward(new_states)

            all_agents_new_actions.append(new_pi)
            mu_states = T.tensor(actor_states[agent_idx], 
                                 dtype=T.float).to(device)
            pi = agent.actor.forward(mu_states)
            all_agents_new_mu_actions.append(pi)
            old_agents_actions.append(actions[agent_idx])

        new_actions = T.cat([acts for acts in all_agents_new_actions], dim=1)
        mu = T.cat([acts for acts in all_agents_new_mu_actions], dim=1)
        old_actions = T.cat([acts for acts in old_agents_actions],dim=1)

        for agent_idx, agent in enumerate(self.agents):
            critic_value_ = agent.target_critic.forward(states_, new_actions).flatten()
            critic_value_[dones[:,0]] = 0.0
            critic_value = agent.critic.forward(states, old_actions).flatten()

            target = rewards[:,agent_idx] + agent.gamma*critic_value_
            #target = target.double()
            #critic_value = critic_value.double()
            #critic_loss = F.mse_loss(target, critic_value)
            critic_loss = F.mse_loss(target.double(), critic_value.double())
            agent.critic.optimizer.zero_grad()
            critic_loss.backward(retain_graph=True)
            agent.critic.optimizer.step()

            actor_loss = agent.critic.forward(states, mu).flatten()
            actor_loss = -T.mean(actor_loss)
            agent.actor.optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            agent.actor.optimizer.step()

            agent.update_network_parameters()
            '''