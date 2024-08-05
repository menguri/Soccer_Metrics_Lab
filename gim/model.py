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
import matplotlib.pyplot as plt
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

# 모델 로드
# sarsa_net_loaded = SARSA_Network(state_size, action_size)
# qvalue_lstm_loaded = QValue_LSTM(state_size, action_size, hidden_size, num_layers)

# sarsa_net_loaded.load_state_dict(torch.load('sarsa_net.pth'))
# qvalue_lstm_loaded.load_state_dict(torch.load('qvalue_lstm.pth'))

# sarsa_net_loaded.eval()
# qvalue_lstm_loaded.eval()

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
model_path = "save/gim_result.ckpt"


def checkpoint_model(PATH, EPOCH, model, h_optimizer, a_optimizer):
    ckp_path = f"./ckpt_save/{PATH}/{EPOCH}/model.ckpt"
    torch.save({
                'epoch': EPOCH,
                'model_state_dict': model.state_dict(),
                'home_optimizer_state_dict': h_optimizer.state_dict(),
                'away_optimizer_state_dict': a_optimizer.state_dict(),
                }, ckp_path)   


def create_mini_batches(data, batch_size):
    mini_batches = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
    return mini_batches


class SarsaLSTMAgent(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, lr, gamma, batch_size, memory_size, max_trace_length, output_dim, num_layers):
        super(SarsaLSTMAgent, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.gamma = gamma
        self.max_trace_length = max_trace_length
        self.memory = deque(maxlen=memory_size)
        # 임의로 batch_size를 정해둔 상태
        self.batch_size = len(self.memory) // 5
        self.model = TwoTowerLSTM(state_dim, hidden_dim, action_dim, output_dim, num_layers)   #.cuda()    input_dim, hidden_dim, output_dim, num_layers, embedding_dim, em_size
        # home, away optimizer 따로 구현
        self.home_optimizer = optim.Adam(list(self.model.home_lstm.parameters())+list(self.model.fc1.parameters())+list(self.model.fc2.parameters()), lr=lr)
        self.away_optimizer = optim.Adam(list(self.model.away_lstm.parameters())+list(self.model.fc1.parameters())+list(self.model.fc2.parameters()), lr=lr)
        self.loss_fn = nn.MSELoss()
    
    def store_init(self):
        self.memory = deque(maxlen=memory_size)

    def store_transition(self, state, action, reward, next_state, next_action, team, trace_length):
        self.memory.append((state, action, reward, next_state, next_action, team, trace_length))

    def sample_batch(self):
        # 무작위로 state index 선택, 마지막 state(골 or 게임 종료)는 무조건 포함. | mini_batch_size는 16
        total = create_mini_batches([i for i in range(len(self.memory))], 32)
        total_batch = []

        for minibatch in total:
            batch_m = []

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
                batch_m.append(trace)
            
            total_batch.append(batch_m)

        return total_batch

    
    def calc_q_loss(self, batch, team):
        q_hat_target = [[], []]

        for b in batch:
            if team == 1:
                q_values_a = self.model.home_forward(b[0], b[3])
                next_q_values_a = self.model.home_forward(b[1], b[4])
            else:
                q_values_a = self.model.away_forward(b[0], b[3])
                next_q_values_a = self.model.away_forward(b[1], b[4])   
        
            wandb.log({
                "Home_prob": q_values_a[0][0],
                "Away_prob": q_values_a[0][1],
                "neither_prob": q_values_a[0][2]
                })

            # target = reward_t+1 + q_t+1
            target_q_values = b[2][0][0][team-1] + next_q_values_a[0][team-1]
            q_hat_target[0].append(q_values_a[0][team-1])
            q_hat_target[1].append(target_q_values)

        q_loss = self.loss_fn(torch.stack(q_hat_target[0]), torch.stack(q_hat_target[1]))
        return q_loss


    def update_model(self):
        if len(self.memory) < 1:
            return

        # update through minibatch 
        total_batch = self.sample_batch()
        batch_loss = [[], []]
        for minibatch in total_batch:
            home_batch = []
            away_batch = []
            for trace in minibatch:
                states, actions, rewards, next_states, next_actions, teams, trace_length = [], [], [], [], [], [], []
                for state, action, reward, next_state, next_action, team, tl in trace:
                    states.append(state)
                    next_states.append(next_state)
                    actions.append(action)
                    next_actions.append(next_action)
                    # reward > r_t+1
                    rewards.append(reward)
                    teams.append(team)
                    trace_length.append(tl)
                # Q(s_t+1, a_t+1)를 구할 땐, next_states (max_trace_length=10을 반영 )
                if len(states) < 10:
                    next_states = [states[0]] + next_states
                    next_actions = [actions[0]] + next_actions
                if len(states) == 0:
                    continue

                # 하나의 Time step (s_t, s_t+1, a_t, a_t+1, r_t+1)을 저장
                states = torch.FloatTensor(np.array(states, dtype=np.float32)).view(1, len(states), -1)                                         
                next_states = torch.FloatTensor(np.array(next_states, dtype=np.float32)).view(1, len(next_states), -1)                          
                rewards = torch.FloatTensor(np.array(rewards[-1], dtype=np.float32)).view(1, 1, -1)                                                                                                          
                actions = torch.FloatTensor(np.array(actions, dtype=np.float32)).view(1, len(actions), -1)                                                                                                             
                next_actions = torch.FloatTensor(np.array(next_actions, dtype=np.float32)).view(1, len(next_actions), -1)                                                                                                

                # team도 scale로 인해 home < 0 값을 가지게 됨.
                if trace[0][-2] < 0:
                    home_batch.append([states, next_states, rewards, actions, next_actions])
                else:
                    away_batch.append([states, next_states, rewards, actions, next_actions])


            if len(home_batch) > 0:
                # home loss update
                home_loss = self.calc_q_loss(home_batch, 1)
                self.home_optimizer.zero_grad()
                home_loss.backward()
                self.home_optimizer.step()
                batch_loss[0].append(int(home_loss.detach()))

            if len(away_batch) > 0:
                # away loss update
                away_loss = self.calc_q_loss(away_batch, 2)
                self.away_optimizer.zero_grad()
                away_loss.backward()
                self.away_optimizer.step()
                batch_loss[1].append(int(away_loss.detach()))
            
        return batch_loss
    

    def predict(self, game):
        y = []
        episode = os.listdir(DATA_STORE + f'/{game}')
        for epi in episode:
            # load data
            s = np.load(DATA_STORE + f'/{game}/{epi}/state.npy', allow_pickle=True)
            a = np.load(DATA_STORE + f'/{game}/{epi}/action.npy', allow_pickle=True)
            # trace_length > possesions
            trace_length = 0
            for t in range(len(s)):
                if trace_length > 10:
                    trace_length = 10
                state = s[t-trace_length:t+1]
                action = a[t-trace_length:t+1]
                action = [np.array(ac) for ac in action]
                state_t = torch.FloatTensor(np.array(state, dtype=np.float32)).view(1, len(state), -1)                                                                                                                                             
                action_t = torch.FloatTensor(np.array(action, dtype=np.float32)).view(1, len(action), -1) 
                if s[t][-1] < 0:
                    output = self.model.home_forward(state_t, action_t)
                else:
                    output = self.model.away_forward(state_t, action_t)

                y.append(output.detach()[0])
                
                # 마지막 state는 trace_length 업데이트 하지 않는다.
                if t+1 == len(s):
                    break

                if s[t][-1] == s[t+1][-1]:
                    trace_length += 1
                else:
                    trace_length = 0
        return y


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

    plt.show()
    plt.savefig(f'./plot/{id}_{epoch}_plot.jpg')