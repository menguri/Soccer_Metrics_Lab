import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


# 하나의 GOAL 이벤트를 학습하며, 골 이벤트 중간의 교대하며 일어나는 Home, Away 선수의 플레이를 각각 다른 LSTM에 넣어주는 개념
# 들어오는 것은 => 10개의 features를 담고 trace length만큼의 seq를 담은 states


# Define the Two-Tower LSTM model for Q-value estimation
class TwoTowerLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, embedding_dim, em_size):
        super(TwoTowerLSTM, self).__init__()
        # hidden staets, hidden layers => 256
        self.hidden_dim = hidden_dim
        # multi-layer LSTM => 2
        self.num_layers = num_layers

        # Home team LSTM
        self.home_lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.home_embedding = nn.Embedding(em_size, embedding_dim)
        # Away team LSTM
        self.away_lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.away_embedding = nn.Embedding(em_size, embedding_dim)

        # Last hiddent layers and softmax
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax()


    def get_q(self, seq, t):
        # Initialize hidden state and cell state for home LSTM
        h0 = torch.zeros(self.num_layers, seq.size(0), self.hidden_dim).to(seq.device)
        c0 = torch.zeros(self.num_layers, seq.size(0), self.hidden_dim).to(seq.device)
 
        # identifier - Home = 1 | Away = 2
        if t == 1:
            # Forward propagate LSTM for home team
            out_home, _ = self.home_lstm(seq, (h0, c0))
            out_home = out_home[:, -1, :]  # Get the last time step's output
            out = self.home_embedding(out_home)
        else:
            out_away, _ = self.away_lstm(seq, (h0, c0))
            out_away = out_away[:, -1, :]  # Get the last time step's output
            out = self.away_embedding(out_away)

        # hidden state + softmax
        output = self.fc(out)
        output = self.softmax(output)

        return output
