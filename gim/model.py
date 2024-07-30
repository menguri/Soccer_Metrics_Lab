# -*- coding: utf-8 -*-
import os
import math
from tqdm import tqdm
from collections import deque

import numpy as np
import pandas as pd
import ast
import random as ran
import datetime
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objects as go
import plotly.express as px
from configuration import learning_rate, batch_size, feature_number, \
                          gamma, num_layers, output_dim, hidden_dim,input_dim, iterate_num, FEATURE_TYPE, MODEL_TYPE, MAX_TRACE_LENGTH, save_mother_dir
from nn.two_tower_lstm import TwoTowerLSTM
import torch
import torch.nn as nn
import torch.optim as optim
from skimage.transform import resize
from skimage.color import rgb2gray
plt.ion()



# Log, Network, Data_Store 경로 지정
LOG_DIR = save_mother_dir + "/models/log_train_feature" + str(
    FEATURE_TYPE) + "_batch" + str(
    batch_size) + "_iterate" + str(
    iterate_num ) + "_lr" + str(
    learning_rate) + "_" + str(MODEL_TYPE) + "_MaxTL" + str(MAX_TRACE_LENGTH)
SAVED_NETWORK = save_mother_dir + "/models/network_train_feature" + str(
    FEATURE_TYPE) + "_batch" + str(
    batch_size) + "_iterate" + str(
    iterate_num ) + "_lr" + str(
    learning_rate) + "_" + str(MODEL_TYPE) + "_MaxTL" + str(MAX_TRACE_LENGTH)
DATA_STORE = "./datastore"

DIR_GAMES_ALL = os.listdir(DATA_STORE)
number_of_total_game = len(DIR_GAMES_ALL)

model_path = "save/gim_result.ckpt"



def train_minibatch(tmodel, mini_batch):
    '''미니배치로 가져온 sample데이터로 네트워크 학습

    Args:
        tmodel(object): 메인 네트워크
        minibatch: replay_memory에서 MINIBATCH 개수만큼 랜덤 sampling 해온 값

    Note:
        replay_memory에서 꺼내온 값으로 메인 네트워크를 학습
    '''
    mini_batch = np.array(mini_batch).transpose()

    history = np.stack(mini_batch[0], axis=0)

    states = np.float32(history[:, :, :, :4])
    actions = list(mini_batch[1])
    rewards = list(mini_batch[2])
    next_rewards = list(mini_batch[4])
    next_actions = list(mini_batch[5])
    next_states = np.float32(history[:, :, :, 1:])
    dones = mini_batch[6]

    # bool to binary
    dones = dones.astype(int)

    Q0 = tmodel.get_q([states, actions])
    Q1 = tmodel.get_q([next_states, next_actions])

    # loss 값 구하기
    loss = next_rewards + Q1 - Q0

    # loss로 tmodel 학습
    


# SARSA 기반 Main() 함수 구현을 통해 학습 실행
def main():

    # Agent 정의
    tmodel = TwoTowerLSTM() 

    # epoch 설정
    epoch = 0

    # 하이퍼파라미터
    alpha = 0.5
    epsilon = 0.1

    epoch_score, epoch_Q = deque(), deque()
    average_Q, average_reward = deque(), deque()
    
    # Train agent during 100 epoch
    while epoch <= 200:
        epoch += 1
        # episode = num(goal sequence) | "We divide a soccer game into goal-scoring episodes"
        for game in DIR_GAMES_ALL:
            episode = os.listdir(DATA_STORE + f'/{game}')
            for epi in episode:
                # load data
                s = np.load(DATA_STORE + f'/{game}/{epi}/state.npy')
                r = np.load(DATA_STORE + f'/{game}/{epi}/reward.npy')
                a = np.load(DATA_STORE + f'/{game}/{epi}/action.npy')

                # Minibatch(하지만 possession 고려해서 잘라야 한다.) & Train
                MINIBATCH_SIZE = 0
                seq = [s,r,a]
                minibatch = ran.sample(seq, MINIBATCH_SIZE)
                train_minibatch(tmodel, minibatch)

        print("Epoch:{0:6d}".format(episode))


if __name__ == "__main__":
    main()