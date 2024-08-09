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
from time import time, sleep
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objects as go
import plotly.express as px
from configuration import state_dim, action_dim, hidden_dim, lr, gamma, batch_size, memory_size, max_trace_length, output_dim, num_layers
from nn.two_tower_lstm import TwoTowerLSTM
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from skimage.transform import resize
from skimage.color import rgb2gray
plt.ion()

# 모델 불러오기



# 모델 예측
game_id = 7529
DATA_STORE = "./datastore"

for epi in [1,2,3,4]:
    dir_game = f'{game_id}_{epi}'
    # load data
    state = sio.loadmat(DATA_STORE + "/" + dir_game + "/" + 'rnn_input')
    trace_length = sio.loadmat(DATA_STORE + "/" + dir_game + "/" + 'trace')
    game += s.tolist()
plot_qvalue_goal(np.array(game), y, game_id, epoch)


def plot_qvalue_goal(game, y, id, epoch=0):
    y = np.array(y)
    gtr = game[:,0]
    gd = game[:,4]
    
    fig, ax1 = plt.subplots(figsize=(15, 10))
    ax1.set_xlabel('GTR')
    ax1.set_ylabel('Prob')
    line1 = ax1.plot(gtr, y[:,0], 'red', label='Home')
    line2 = ax1.plot(gtr, y[:,1], 'blue', label='Away')
    line3 = ax1.plot(gtr, y[:,2], 'yellow', label='Neither')

    ax2 = ax1.twinx()
    ax2.set_ylabel('GD')
    line4 = ax2.plot(gtr, gd, 'deeppink', label='Goal Difference', linestyle="--")

    lines = line1 + line2 + line3 + line4
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right')

    plt.savefig(f'./plot/goal_prob/{id}_{epoch}_plot.jpg')