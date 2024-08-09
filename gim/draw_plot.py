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
from configuration import MODEL_TYPE, MAX_TRACE_LENGTH, FEATURE_NUMBER, BATCH_SIZE, GAMMA, H_SIZE, \
    model_train_continue, FEATURE_TYPE, ITERATE_NUM, learning_rate, SPORT, save_mother_dir, hidden_dim, lr, gamma, batch_size, memory_size, max_trace_length, output_dim, num_layers
from nn.td_two_tower_lstm import TD_Prediction_TT_Embed
from utils import handle_trace_length, get_together_training_batch, compromise_state_trace_length
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import scipy.io as sio
import torch.optim as optim
from skimage.transform import resize
from skimage.color import rgb2gray
plt.ion()


SAVED_NETWORK = str(os.getcwd()) + save_mother_dir + "/models/hybrid_sl_saved_NN/Scale-three-cut_together_saved_networks_feature" + str(
    FEATURE_TYPE) + "_batch" + str(
    BATCH_SIZE) + "_iterate" + str(
    ITERATE_NUM) + "_lr" + str(
    learning_rate) + "_" + str(MODEL_TYPE) + "_MaxTL" + str(MAX_TRACE_LENGTH)


# 모델 불러오기
# loading network
model = TD_Prediction_TT_Embed(FEATURE_NUMBER, hidden_dim, MAX_TRACE_LENGTH, learning_rate)
check_path = os.path.join(SAVED_NETWORK, f"{SPORT}-game-{1000}.pt")
checkpoint = torch.load(check_path)  # Load checkpoint
model.load_state_dict(checkpoint['model_state_dict'])
model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
game_starting_point = 0


# 데이터 로드
game_id = 7525
DATA_STORE = "./datastore"
game_list = []
trace_length_list = []
reward_list = []

for epi in [1,2,3,4,5,6]:
    dir_game = f'{game_id}_{epi}'
    # load data
    state = sio.loadmat(DATA_STORE + "/" + dir_game + "/" + 'rnn_input')
    reward = sio.loadmat(DATA_STORE + "/" + dir_game + "/" + 'rnn_reward')
    trace_length = sio.loadmat(DATA_STORE + "/" + dir_game + "/" + 'trace')
    game_list += (state['data'])[0].tolist()
    reward_list += (reward['data'])[0].tolist()
    trace_length_list += (trace_length['data'])[0].tolist()

GTR = [state[-1][0] for state in game_list]
GD = [state[-1][4] for state in game_list]
home_away_indicator = [1 if state[-1][-1] > 1 else 0 for state in game_list]

# Prediction
state_trace_length, state_input, reward = compromise_state_trace_length(trace_length_list, game_list, reward_list, MAX_TRACE_LENGTH)

# get the batch variables
y_batch = []

# Target 값 계산
trace_batch_tensor = torch.tensor(state_trace_length, dtype=torch.int32)
s_batch_tensor = torch.tensor(state_input, dtype=torch.float32)
home_away_indicator_tensor = torch.tensor(home_away_indicator, dtype=torch.bool)
# 모델을 평가 모드로 전환
model.eval()
# forward를 통해 출력 계산
with torch.no_grad():
    outputs_t0 = model.forward(s_batch_tensor, trace_batch_tensor, home_away_indicator_tensor)
# 필요 시 numpy 배열로 변환 (TensorFlow의 sess.run()과 동일한 역할)
readout_t1_batch = outputs_t0.numpy()


def plot_qvalue_goal(GTR, GD, state, y, game_id):
    print(f"len gtr : {len(GTR)}")
    print(f"len GD : {len(GD)}")
    print(f"len y : {len(y)}")
    gtr = GTR
    gd = GD
    
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

    plt.gca().invert_xaxis()  # x축 반전
    plt.title(f'{game_id} Goal Probability')
    plt.savefig(f'./plot/goal_prob/{game_id}_plot.jpg')


plot_qvalue_goal(GTR, GD, np.array(game_list), readout_t1_batch, game_id)