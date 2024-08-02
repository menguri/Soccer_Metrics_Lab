# -*- coding: utf-8 -*-
import os
import math
from tqdm import tqdm
from collections import deque
import random
import numpy as np
import pandas as pd
import ast
import random as ran
import datetime
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objects as go
import plotly.express as px
from configuration import state_dim, action_dim, hidden_dim, lr, gamma, epsilon, batch_size, memory_size, max_trace_length, output_dim, num_layers, embedding_dim, em_size
from nn.two_tower_lstm import TwoTowerLSTM
import torch
import torch.nn as nn
import torch.optim as optim
from skimage.transform import resize
from skimage.color import rgb2gray
plt.ion()


# Log, Network, Data_Store 경로 지정
# LOG_DIR = save_mother_dir + "/models/log_train_feature" + str(
#     FEATURE_TYPE) + "_batch" + str(
#     batch_size) + "_iterate" + str(
#     iterate_num ) + "_lr" + str(
#     learning_rate) + "_" + str(MODEL_TYPE) + "_MaxTL" + str(MAX_TRACE_LENGTH)
# SAVED_NETWORK = save_mother_dir + "/models/network_train_feature" + str(
#     FEATURE_TYPE) + "_batch" + str(
#     batch_size) + "_iterate" + str(
#     iterate_num ) + "_lr" + str(
#     learning_rate) + "_" + str(MODEL_TYPE) + "_MaxTL" + str(MAX_TRACE_LENGTH)
DATA_STORE = "./datastore"

DIR_GAMES_ALL = os.listdir(DATA_STORE)
number_of_total_game = len(DIR_GAMES_ALL)

model_path = "save/gim_result.ckpt"


class SarsaLSTMAgent:
    def __init__(self, state_dim, action_dim, hidden_dim, lr, gamma, epsilon, batch_size, memory_size, max_trace_length, output_dim, num_layers, embedding_dim, em_size):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.max_trace_length = max_trace_length
        self.memory = deque(maxlen=memory_size)
        # 임의로 batch_size를 정해둔 상태
        self.batch_size = len(self.memory) // 5
        self.model = TwoTowerLSTM(state_dim, hidden_dim, action_dim, output_dim, num_layers, embedding_dim, em_size)   #.cuda()    input_dim, hidden_dim, output_dim, num_layers, embedding_dim, em_size
        # home, away optimizer 따로 구현
        self.home_optimizer = optim.Adam(list(self.model.home_lstm.parameters())+list(self.model.fc1.parameters())+list(self.model.fc2.parameters()), lr=lr)
        self.away_optimizer = optim.Adam(list(self.model.away_lstm.parameters())+list(self.model.fc1.parameters())+list(self.model.fc2.parameters()), lr=lr)
        self.loss_fn = nn.MSELoss()
    
    def store_init(self):
        self.memory = deque(maxlen=memory_size)

    def store_transition(self, state, action, reward, next_state, next_action, team, trace_length):
        self.memory.append((state, action, reward, next_state, next_action, team, trace_length))

    def sample_batch(self):
        # 무작위로 state index 선택, 마지막 state(-1)는 무조건 포함. | batch_size는 epi 길이의 1/5로 임의 정함.
        minibatch = random.sample([i for i in range(len(self.memory))], len(self.memory)//5) + [-1]
        batch = []

        # trace_length 고려해서 trace를 구성
        for play in minibatch:
            trace_length = self.memory[play][-1]
            if trace_length > self.max_trace_length :
                trace_length = 10
            trace = list(self.memory)[play - trace_length + 1:play+1]
            if play == -1:
                trace = list(self.memory)[play - trace_length + 1:]
            if len(trace) == 0:
                print(f"This trace has some errors : {self.memory[play]}")
            batch.append(trace)
        
        return batch
    
    def calc_q_loss(self, states, next_states, actions, next_actions, rewards, t):
        if t == 1:
            q_values_a = self.model.home_forward(states, actions, t) #.gather(2, actions)
            next_q_values_a = self.model.home_forward(next_states, next_actions, t) #.gather(2, next_actions)
        else:
            q_values_a = self.model.away_forward(states, actions, t) #.gather(2, actions)
            next_q_values_a = self.model.away_forward(next_states, next_actions, t) #.gather(2, next_actions)  
        
        print(f"team : {t}")
        print(f"reward : {rewards} / reward : {rewards[0][0][t-1]}")
        print(f"q_values_a : {q_values_a}")
        print(f"next_q_values_a : {next_q_values_a}")
        target_q_values = rewards[0][0][t-1] + next_q_values_a[0][t-1]
        q_loss = self.loss_fn(q_values_a[0][t-1], target_q_values.detach())

        return q_loss


    def update_model(self):
        if len(self.memory) < 1:
            return

        minibatch = self.sample_batch()
        home_loss = []
        away_loss = []
        for trace in minibatch:
            states, actions, rewards, next_states, next_actions, teams, trace_length = [], [], [], [], [], [], []
            for state, action, reward, next_state, next_action, team, tl in trace:
                states.append(state)
                actions.append(action)
                # reward = r_t+1
                rewards.append(reward)
                next_states.append(next_state)
                next_actions.append(next_action)
                teams.append(team)
                trace_length.append(tl)
            # Q(s_t+1, a_t+1)를 구할 땐, next_states (max_trace_length=10을 반영 )
            if len(states) < 10:
                next_states = [states[0]] + next_states
                next_actions = [actions[0]] + next_actions
            if len(states) == 0:
                continue

            # trace_length가 seq마다 다르기 때문에 한 개의 T마다 update.
            states = torch.FloatTensor(np.array(states, dtype=np.float32)).view(1, len(states), -1)                                         
            next_states = torch.FloatTensor(np.array(next_states, dtype=np.float32)).view(1, len(next_states), -1)                          
            rewards = torch.FloatTensor(np.array(rewards[-1], dtype=np.float32)).view(1, 1, -1)                                                                                                          
            actions = torch.FloatTensor(np.array(actions, dtype=np.float32)).view(1, len(actions), -1)                                                                                                             
            next_actions = torch.FloatTensor(np.array(next_actions, dtype=np.float32)).view(1, len(next_actions), -1)                                                                                                

            loss = self.calc_q_loss(states, next_states, actions, next_actions, rewards, trace[0][-2])

            if trace[0][-2] == 1:
                # home loss update
                self.home_optimizer.zero_grad()
                loss.backward()
                self.home_optimizer.step()
                home_loss.append(loss.item())

            else:
                # away loss update
                self.away_optimizer.zero_grad()
                loss.backward()
                self.away_optimizer.step()
                away_loss.append(loss.item())

        return home_loss, away_loss
