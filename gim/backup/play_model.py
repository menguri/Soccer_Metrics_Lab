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
from time import time, sleep
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
from model import SarsaLSTMAgent, plot_qvalue_goal, checkpoint_model

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
    while epoch <= 30:
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
                    # 마지막인 경우와 state가 1개 밖에 없는 경우(대체로 freekick) 
                    if t+1 == len(s) or len(s) < 2:
                        break
                    # t's team == t+1's team
                    if s[t][-1] == s[t+1][-1]:
                        agent.store_transition(s[t], a[t], r[t+1], s[t+1], a[t+1], s[t][-1], trace_length)
                        trace_length += 1
                    else:
                        # t의 team과 t+1의 team이 다른 경우, trace_length 초기화
                        trace_length = 1

                # agent update (calc_q_loss)
                batch_loss = agent.update_model()
                try:
                    wandb.log({"Home Training loss": statistics.mean(batch_loss[0])})
                    wandb.log({"Away Training loss": statistics.mean(batch_loss[1])})
                    print(f"{game} game's {epi} episode >> Home loss: {statistics.mean(batch_loss[0])} | Away loss : {statistics.mean(batch_loss[1])}")
                except:
                    pass

            # game마다 memory init
            agent.store_init()
            print(f"------------------ Epoch: {epoch} / Game: {game} ------------------")


        # epoch 1 마다 저장 및 plot 그리기
        if epoch % 1 == 0:
            checkpoint_model(f"{lr}lr_{batch_size}batchsize", epoch, agent, agent.home_optimizer, agent.away_optimizer)
            # Plot
            # 모델 예측
            game_id = 7525
            y = agent.predict(7525)
            game = []
            episode = os.listdir(DATA_STORE + f'/{7525}')
            for epi in episode:
                # load data
                s = np.load(DATA_STORE + f'/{7525}/{epi}/state.npy', allow_pickle=True)
                game += s.tolist()
            plot_qvalue_goal(np.array(game), y, game_id, epoch)
            sleep(2)

        print("------------------ Epoch: {} ------------------".format(epoch))

    
    # 모델 저장
    path = './last_model.pth'
    torch.save(agent.state_dict(), path)


if __name__ == "__main__":
    main()



# plot 그리기
plot = False

if plot == True:
    # Plot
    path = './last_model.pth'
    agent = SarsaLSTMAgent(state_dim, action_dim, hidden_dim, lr, gamma, batch_size, memory_size, max_trace_length, output_dim, num_layers)
    agent.load_state_dict(torch.load(path))
    agent.eval()
    # 모델 예측
    game_id = 7525
    y = agent.predict(7525)
    game = []
    episode = os.listdir(DATA_STORE + f'/{7525}')
    for epi in episode:
        # load data
        s = np.load(DATA_STORE + f'/{7525}/{epi}/state.npy', allow_pickle=True)
        game += s.tolist()
    plot_qvalue_goal(np.array(game), y, id)