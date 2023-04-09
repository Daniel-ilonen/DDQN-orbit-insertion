import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
import collections
from random import randint
DEVICE = torch.device("cpu")
class DQNagent(torch.nn.Module):
    def __init__(self, params,state_space,action_space):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.memory_size=params['memory_size']
        self.batch_size=params['batch_size']
        
        self.gamma=params['gamma']
        self.epsilon_decay=params['epsilon_decay']
        self.epsilon_min=params['epsilon_min']
        
        self.fc1 = nn.Linear(self.state_space, params['first_layer_size'])
        self.fc2 = nn.Linear(params['first_layer_size'], params['second_layer_size'])
        self.fc3 = nn.Linear(params['second_layer_size'], self.action_space)

        self.optimizer = optim.Adam(self.parameters(), lr= params['learning_rate'])
        self.loss = nn.MSELoss()
        self.to(DEVICE)
        self.memory = collections.deque(maxlen=self.memory_size)
        self.exploration_rate = params['epsilon_max']
        
        self.mem_count = 0
        self.train()
        self.states = np.zeros((self.memory_size, self.state_space),dtype=np.float32)
        self.actions = np.zeros(self.memory_size, dtype=np.int64)
        self.rewards = np.zeros(self.memory_size, dtype=np.float32)
        self.states_ = np.zeros((self.memory_size, self.state_space),dtype=np.float32)
        self.dones = np.zeros(self.memory_size, dtype=np.bool)
        
        if params["train"] == False:
            self.exploration_rate=self.epsilon_min
        
        self.load_weights = params['load_weights']
        self.weights = params['weights_path']

          
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


    def remember(self, state, action, reward, state_, done):
        mem_index = self.mem_count % self.memory_size
        
        self.states[mem_index]  = state
       # print(self.states[mem_index][0])
        self.actions[mem_index] = action
        self.rewards[mem_index] = reward
        self.states_[mem_index] = state_
        self.dones[mem_index] =  1 - done

        self.mem_count += 1
    
    def sample(self):
        MEM_MAX = min(self.mem_count, self.memory_size)
        batch_indices = np.random.choice(MEM_MAX, self.batch_size, replace=True)
        
        states  = self.states[batch_indices]
        #print(states[0])
        actions = self.actions[batch_indices]
        rewards = self.rewards[batch_indices]
        states_ = self.states_[batch_indices]
        dones   = self.dones[batch_indices]

        return states, actions, rewards, states_, dones


    def act(self, observation):
        if random.random() < self.exploration_rate:
            return np.argmax(np.eye(self.action_space)[randint(0,self.action_space-1)])
        
        state = torch.tensor(observation).float().detach()
        state = state.to(DEVICE)
        state = state.unsqueeze(0)
        q_values = self(state)
        return torch.argmax(q_values).item()
    def update_parameters(self, target_model):
        target_model.load_state_dict(self.state_dict())
        target_model.exploration_rate=self.exploration_rate
    
    def replay(self,target):
        if self.mem_count < self.batch_size:
            return
        
        #Sample the memory space
        states, actions, rewards, states_, dones = self.sample()
        
        states = torch.tensor(states , dtype=torch.float32).to(DEVICE)
        actions = torch.tensor(actions, dtype=torch.long).to(DEVICE)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(DEVICE)
        states_ = torch.tensor(states_, dtype=torch.float32).to(DEVICE)
        dones = torch.tensor(dones, dtype=torch.bool).to(DEVICE)
        batch_indices = np.arange(self.batch_size, dtype=np.int64)

        #Define Q-values
        q_values = self(states)       
        next_q_values = self(states_)
        next_q_state_values = target(states_)
        
        q_value = q_values[batch_indices, actions] 
        next_q_value = next_q_state_values[batch_indices,torch.max(next_q_values, 1)[1]]
        
        
        q_target = rewards + self.gamma * next_q_value * dones

        loss = self.loss(q_target, q_value)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.exploration_rate *= self.epsilon_decay
        self.exploration_rate = max(self.epsilon_min, self.exploration_rate)

    def returning_epsilon(self):
        return self.exploration_rate
