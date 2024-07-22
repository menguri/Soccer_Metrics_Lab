# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import tqdm
from scipy import signal


# GTR: game time remaining | ED: event duration
def game_remain(gamestates, duration_drop=True):
    period_sec = []
    end_dur = 0
    ## GTR 구하기 : 경기 총 시간(s) - 해당 time_seconds
    for id in [1,2,3,4,5]:
        try: 
            period_sec.append(gamestates[gamestates['period_id']==id]['time_seconds'].iloc[-1])
            end_dur = gamestates[gamestates['period_id']==id]['duration'].iloc[-1]
        except: break

    gtr = []
    total_sec = sum(period_sec) + end_dur

    for id in range(0, len(period_sec)): 
        gtr += [total_sec - sum(period_sec[:id]) - time for time in list(gamestates[gamestates['period_id'] == id+1]['time_seconds'])]
    gamestates['GTR'] = np.round(gtr, 4)  
    
    ## ED 구하기
    if duration_drop == True:
        gamestates.dropna(subset='duration', axis=0, inplace=True)
    else: 
        gamestates['new_dur'] = list(gamestates['time_seconds'].diff())[1:] + [0]
        gamestates['ED'] = np.where(pd.notnull(gamestates['duration'])==True,
                                          gamestates['duration'], gamestates['new_dur'])
        
        gamestates.drop('new_dur', axis=1, inplace=True)

    return gamestates


# MP: manpower(For the feature manpower situation, negative values indicate short-handed, positive values indicate power play) 
# GD: goal difference
def goal_difference(gamestates):
    goal = [0, 0]  # [home, away]
    
    return gamestates


# Action: one-hot representation
def onehot_action(dataset):
    type_uni = dataset['type_name'].unique()
    type_to_index = {type_ : index for index, type_ in enumerate(type_uni)}
    type_list = []

    for idx, action in tqdm.tqdm(dataset.iterrows(), desc="one-hot encoding"):
        one_hot_vector = [0]*(len(type_to_index))
        index = type_to_index[action.type_name]
        one_hot_vector[index] = 1
        type_list.append(one_hot_vector)
    
    dataset['Action'] = type_list
    return dataset


# Team: Home, Away
def get_team(dataset):
    team_list = []
    for idx, action in tqdm.tqdm(dataset.iterrows(), desc="Team discrete"):
        if action.team_id == action.home_team_id:
            team_list.append('Home')
        else:
            team_list.append('Away')
    dataset['T'] = team_list
    return dataset


# Angle: between ball and goal | Velocity: (end_location - start_location) / time
def get_angle_velocity(dataset, field_dims=(100, 100), window=2, polyorder=1):

    angle_bet = []
    velocity_bet = []

    # 축구장 규격은 100 X 100 | 축구 골대 = (100, 50)
    for idx, action in tqdm.tqdm(dataset.iterrows(), desc="Calculating angle, velocity"):

        # ball의 x, y 좌표
        ball_start_location = np.array([action.start_x, action.start_y])
        ball_start_coo = ball_start_location.reshape(-1, 2)
        ball_end_location = np.array([action.end_x, action.end_y])
        ball_end_coo = ball_end_location.reshape(-1, 2)

        # 골대 x, y 좌표
        goal_coo = np.array([100, 50]).reshape(-1, 2)

        # ball start - ball end | ball end - goal
        a = ball_end_coo - ball_start_coo
        b = goal_coo - ball_end_coo

        # angle between ball and goal
        angle = np.arccos(np.clip(np.sum(a * b, axis=1) / (np.linalg.norm(a) * np.linalg.norm(b, axis=1) + np.finfo(float).eps), -1, 1))
        angle_bet.append(angle[0])

        # velocity: (end_location - start_location)/duration
        x = np.array([ball_start_location[0], ball_end_location[0]])
        y = np.array([ball_start_location[1], ball_end_location[1]])
        x = pd.Series(signal.savgol_filter(x, window_length=window, polyorder=polyorder))
        y = pd.Series(signal.savgol_filter(y, window_length=window, polyorder=polyorder))

        try:
            vx = list(x.diff())[1] / action.ED
            vy = list(y.diff())[1] / action.ED
        except:
            # goal
            vx = 0
            vy = 0
        
        velocity_bet.append(np.array([vx, vy]))

        if idx == 0:
            print('첫 행 계산 과정')
            print(f'ball_start: {ball_start_coo}')
            print(f'ball_end: {ball_end_coo}')
            print(f'goal_location: {goal_coo}')
            print(f'end - start: {a}')
            print(f'goal - end: {b}')
            print(f'angle: {angle}')
            print(f'X: {x}')
            print(f'Y: {y}')
            print(f'VX: {vx}')
            print(f'VY: {vy}')
    
    dataset['Angle'] = angle_bet
    dataset['Velocity'] = velocity_bet
    return dataset


# Reward: [home, away, neither]
def get_reward(dataset):

    reward = [[0,0,0] for i in range(len(dataset))]

    goal_idx = list(dataset[dataset['type_name'] == 'goal'].index)

    for idx in goal_idx:
        g = dataset.loc[idx]
        if g.team_id == g.home_team_id:
            reward[idx] = [1,0,0]
        else:
            reward[idx] = [0,1,0]

    dataset['Reward'] = reward
    return dataset


