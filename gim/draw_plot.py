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


MODEL_VERSION = 7300


def plot_qvalue_goal(GTR, GD, SHOT_TRY, state, y, game_id, ITER):
    print(f"len gtr : {len(GTR)}")
    print(f"len GD : {len(GD)}")
    print(f"len y : {len(y)}")
    
    fig, ax1 = plt.subplots(figsize=(15, 10))
    ax1.set_xlabel('GTR')
    ax1.set_ylabel('Prob')
    line1 = ax1.plot(GTR, y[:,0], 'red', label='Home', linestyle="--")
    line2 = ax1.plot(GTR, y[:,1], 'blue', label='Away', linestyle="--")
    line3 = ax1.plot(GTR, y[:,2], 'yellow', label='Neither', linestyle="--")

    ax2 = ax1.twinx()
    ax2.set_ylabel('GD')
    line4 = ax2.plot(GTR, GD, 'deeppink', label='Goal Difference')

    lines = line1 + line2 + line3 + line4
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right')

    plt.gca().invert_xaxis()  # x축 반전

    # 샷이 일어난 순간을 수직선으로 표시
    for shot in SHOT_TRY:
        color = 'red' if shot[1] == 1 else 'blue'
        shot_team = 'H' if shot[1] == 1 else 'A'
        plt.vlines(x=shot[0], ymin=-0.001, ymax=0.003, color=color, linewidth=2, alpha=0.7)
        plt.text(shot[0], 0.0035, shot_team, color=color, fontsize=12, ha='center')

    plt.title(f'{game_id} Goal Probability')
    plt.savefig(f'./plot/goal_prob/{game_id}_{ITER}.jpg')



def game_plot(FEATURE_NUMBER, hidden_dim, MAX_TRACE_LENGTH, learning_rate, SAVED_NETWORK, SPORT, MODEL_VERSION, GAME_ID):
    # loading network
    # 모델 불러오기
    model = TD_Prediction_TT_Embed(FEATURE_NUMBER, hidden_dim, MAX_TRACE_LENGTH, learning_rate)
    check_path = os.path.join(SAVED_NETWORK, f"{SPORT}-game-{MODEL_VERSION}.pt")
    checkpoint = torch.load(check_path)  # Load checkpoint
    model.load_state_dict(checkpoint['model_state_dict'])
    model.optimizer_home.load_state_dict(checkpoint['home_optimizer_state_dict'])
    model.optimizer_away.load_state_dict(checkpoint['away_optimizer_state_dict'])
    check_point_game_number = checkpoint['check_point_game_number']

    # 데이터 로드
    game_id = GAME_ID

    DATA_STORE = "./datastore"
    game_list = []
    trace_length_list = []
    reward_list = []

    # load data
    dir_game = f'{game_id}'
    state = sio.loadmat(DATA_STORE + "/" + dir_game + "/" + 'rnn_input')
    reward = sio.loadmat(DATA_STORE + "/" + dir_game + "/" + 'rnn_reward')
    trace_length = sio.loadmat(DATA_STORE + "/" + dir_game + "/" + 'trace')
    game_list += (state['data'])[0].tolist()
    reward_list += (reward['data'])[0].tolist()
    trace_length_list += (trace_length['data'])[0].tolist()
    
    
    GTR = []
    GD = []
    SHOT_TRY = []
    for state in game_list:
        GTR.append(state[-1][0])
        GD.append(state[-1][4])
        if state[-1][22] == 1 or state[-1][29] == 1 or state[-1][32] == 1:
            SHOT_TRY.append([state[-1][0], round(state[-1][10])])
        if state[-1][31] == 1 :
            SHOT_TRY.append([state[-1][0], -round(state[-1][10])])        


    home_away_indicator = [1 if state[-1][10] > 1 else 0 for state in game_list]
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
    plot_qvalue_goal(GTR, GD, SHOT_TRY, np.array(game_list), readout_t1_batch, game_id, check_point_game_number)


# Plot 그리기
MODEL_VERSION = 1920
DATA_STORE = "./datastore"
DIR_GAMES_ALL = os.listdir(DATA_STORE)

for GAME_ID in DIR_GAMES_ALL: 
    game_plot(FEATURE_NUMBER, hidden_dim, MAX_TRACE_LENGTH, learning_rate, SAVED_NETWORK, SPORT, MODEL_VERSION, GAME_ID)