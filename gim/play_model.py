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
from configuration import state_dim, action_dim, hidden_dim, lr, gamma, batch_size, memory_size, max_trace_length, output_dim, num_layers
from nn.two_tower_lstm import TwoTowerLSTM
import torch
import torch.nn as nn
import torch.optim as optim
from skimage.transform import resize
from skimage.color import rgb2gray
import statistics
plt.ion()
from model import SarsaLSTMAgent, plot_qvalue_goal

# Experiment tracking : Wandb
import wandb
wandb.init(project='Sarsa with LSTM')
wandb.run.name = 'Soccer Sarsa'
wandb.run.save()
args = {
    "learning_rate": lr,
    "gamma": gamma,
    "buffer_limit": memory_size
}
wandb.config.update(args)


DATA_STORE = "./datastore"

DIR_GAMES_ALL = os.listdir(DATA_STORE)
number_of_total_game = len(DIR_GAMES_ALL)



# SARSA 기반 Main() 함수 구현을 통해 학습 실행
def main():
    agent = SarsaLSTMAgent(state_dim, action_dim, hidden_dim, lr, gamma, batch_size, memory_size, max_trace_length, output_dim, num_layers)
    epoch = 0
    # Train agent during 30 epoch
    while epoch <= 1:
        epoch += 1
        # episode = num(goal sequence) | "We divide a soccer game into goal-scoring episodes"
        for game in DIR_GAMES_ALL:
            episode = os.listdir(DATA_STORE + f'/{game}')
            for epi in episode:
                # load data
                s = np.load(DATA_STORE + f'/{game}/{epi}/state.npy', allow_pickle=True)
                r = np.load(DATA_STORE + f'/{game}/{epi}/reward.npy', allow_pickle=True)
                a = np.load(DATA_STORE + f'/{game}/{epi}/action.npy', allow_pickle=True)
                # Memory -> T(state_t, action_t, reward_t+1, state_t+1, action_t+1, Team, Trace_Length)
                trace_length = 1
                for t in range(len(s)):
                    # t이 마지막인 경우, 종료
                    if sum(r[t]) == 1:
                        break
                    # t's team == t+1's team
                    try:
                        if s[t][-1] == s[t+1][-1]:
                            agent.store_transition(s[t], a[t], r[t+1], s[t+1], a[t+1], s[t][-1], trace_length)
                            trace_length += 1
                        else:
                            # t의 team과 t+1의 team이 다른 경우, trace_length 초기화
                            trace_length = 1
                    except:
                        print(f"s: {s}")
                        print(f"s_t: {s[t]}")

                # agent update (calc_q_loss)
                batch_loss = agent.update_model()
                wandb.log({"Home Training loss": statistics.mean(batch_loss[0])})
                wandb.log({"Away Training loss": statistics.mean(batch_loss[1])})
                print(f"{game} game's {epi} episode >> Home loss: {statistics.mean(batch_loss[0])} | Away loss : {statistics.mean(batch_loss[1])}")

            # game마다 memory init
            agent.store_init()
        print("Epoch: {}".format(epoch))

    
    # 모델 저장
    torch.save(agent.state_dict(), 'sarsa_net.pth')

    # 모델 예측
    y = agent.predict(7525)
    game = []
    episode = os.listdir(DATA_STORE + f'/{7525}')
    for epi in episode:
        # load data
        s = np.load(DATA_STORE + f'/{game}/{epi}/state.npy', allow_pickle=True)
        game.append(s)
    plot_qvalue_goal(np.array(game), y)


if __name__ == "__main__":
    main()
