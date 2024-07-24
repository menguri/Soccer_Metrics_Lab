import os
import math
from tqdm import tqdm

import numpy as np
import pandas as pd
import ast
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objects as go
import plotly.express as px
from configuration import learning_rate, batch_size, Feature_number, \
                          gamma, num_layers, output_dim, hidden_dim,input_dim, Iterate_num 
from nn.two_tower_lstm import TwoTowerLSTM
import torch
import torch.nn as nn
import torch.optim as optim


# SARSA Agent
class SARSA:
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, learning_rate=0.001, gamma=0.99):
        self.model = TwoTowerLSTM(input_dim, hidden_dim, output_dim, num_layers)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.gamma = gamma

    def predict(self, seq, t):
        self.model.eval()
        with torch.no_grad():
            q_ = self.model(seq, t)
        return q_

    def update(self, state, action, reward, next_state, next_action, done, team):
        self.model.train()
        home_seq, away_seq = state
        next_home_seq, next_away_seq = next_state

        q_home, q_away = self.model(home_seq, away_seq)
        q_next_home, q_next_away = self.model(next_home_seq, next_away_seq)

        q_target_home = q_home.clone()
        q_target_away = q_away.clone()

        if team == 'Home':
            q_update = reward + (self.gamma * q_next_home[next_action] * (1 - done))
            q_target_home[action] = q_update
        elif team == 'Away':
            q_update = reward + (self.gamma * q_next_away[next_action] * (1 - done))
            q_target_away[action] = q_update

        loss_home = self.criterion(q_home, q_target_home)
        loss_away = self.criterion(q_away, q_target_away)

        loss = loss_home + loss_away
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()