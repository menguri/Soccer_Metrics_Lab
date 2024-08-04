# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import tqdm
from scipy import signal


# GTR: game time remaining | ED: event duration
def game_remain(gamestates, duration_drop=False):

    ## ED 구하기
    if duration_drop == True:
        gamestates.dropna(subset='duration', axis=0, inplace=True)
    else: 
        gamestates['new_dur'] = list(gamestates['time_seconds'].diff())[1:] + [0]
        gamestates['ED'] = np.where(pd.notnull(gamestates['duration'])==True,
                                          gamestates['duration'], gamestates['new_dur'])
        
        gamestates.drop('new_dur', axis=1, inplace=True)
    

    ## GTR 구하기 : 경기 총 시간(s) - 해당 time_seconds
    period_sec = []
    end_dur = 0
    for id in [1,2,3,4,5]:
        try: 
            period_sec.append(gamestates[gamestates['period_id']==id]['time_seconds'].iloc[-1])
            end_dur = gamestates[gamestates['period_id']==id]['ED'].iloc[-1]
        except: break

    if len(period_sec) < 1:
        print(f"game : {gamestates}")
        return gamestates
    gtr = []
    total_sec = sum(period_sec) + end_dur

    for id in range(0, len(period_sec)): 
        gtr += [round((total_sec - sum(period_sec[:id]) - time)/total_sec * 100, 2) for time in list(gamestates[gamestates['period_id'] == id+1]['time_seconds'])]
    gamestates['GTR'] = np.round(gtr, 4)  

    return gamestates


# MP: manpower(For the feature manpower situation, negative values indicate short-handed, positive values indicate power play)
def is_out(row):
    for e in ["foul_committed", "bad_behaviour"]:
        try:
            if e in row.extra and "card" in row.extra[e] and row.extra[e]["card"]["name"] in ["Second Yellow", "Red Card"] :
                return 1
        except:
            break
    return 0

def get_manpower(dataset, games):
    dataset['red'] = dataset.apply(is_out, axis = 1)
    game = []

    for g_id in tqdm.tqdm(list(games['game_id'].unique()), desc="Calculating manpower..."):
        gamestates = dataset[dataset['game_id'] == g_id].copy()
        home_team_id, away_team_id = gamestates[['home_team_id', 'away_team_id']].iloc[0].values
        power = [11, 11]  # [Home player, Away player]
        mp = []
        for idx, action in gamestates.iterrows():
            if action.red == 1 and action.team_id == home_team_id:
                power[0] -= 1
            elif action.red == 1 and action.team_id == away_team_id:
                power[1] -= 1
            mp.append(power[0] - power[1])

        gamestates['MP'] = mp
        game.append(gamestates)

    dataset = pd.concat(game).sort_values("game_id").reset_index(drop=True)
    return dataset


# GD: goal difference
def goal_difference(gamestates):
    goal = [0, 0]  # [home, away]
    gd_list = []
    for idx, action in gamestates.iterrows():
        gd_list.append(goal[0] - goal[1])
        if (action.type_name == 'goal')&(action['T'] == 1):
            goal[0] += 1
        elif (action.type_name == 'goal')&(action['T'] == 2):
            goal[1] += 1 
    gamestates['GD'] = gd_list
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


# Team: Home(1), Away(2)
def get_team(dataset):
    team_list = []
    for idx, action in tqdm.tqdm(dataset.iterrows(), desc="Team discrete"):
        if action.team_id == action.home_team_id:
            team_list.append(1)
        else:
            team_list.append(2)
    dataset['T'] = team_list
    return dataset


# Angle: between ball and goal | Velocity: (end_location - start_location) / time
def get_angle_velocity(dataset, field_dims=(100, 100), window=2, polyorder=1):

    angle_bet = []
    ball_vx = []
    ball_vy = []

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
        
        ball_vx.append(vx)
        ball_vy.append(vy)

        # if idx == 0:
        #     print('첫 행 계산 과정')
        #     print(f'ball_start: {ball_start_coo}')
        #     print(f'ball_end: {ball_end_coo}')
        #     print(f'goal_location: {goal_coo}')
        #     print(f'end - start: {a}')
        #     print(f'goal - end: {b}')
        #     print(f'angle: {angle}')
        #     print(f'X: {x}')
        #     print(f'Y: {y}')
        #     print(f'VX: {vx}')
        #     print(f'VY: {vy}')
    
    dataset['Angle'] = angle_bet
    dataset['VX'] = ball_vx
    dataset['VY'] = ball_vy
    return dataset


# Reward: [home, away, neither]
def get_reward(dataset, games):

    reward = [[0,0,0] for i in range(len(dataset))]

    # goal -> reward 부여
    goal_idx = list(dataset[dataset['type_name'] == 'goal'].index)
    for idx in goal_idx:
        g = dataset.loc[idx]
        if g.team_id == g.home_team_id:
            reward[idx] = [1,0,0]
        else:
            reward[idx] = [0,1,0]
    
    # Neither -> [0,0,1] 부여  "an extra Neither indicator for the eventuality that neither team scores until the end of a game."
    # 각 period 마지막에 아무것도 없이 끝난다면 neither 부여
    neither_idx = []
    for g_id in list(games['game_id'].unique()):
        gamestates = dataset[dataset['game_id'] == g_id].copy()
        for id in [1,2,3,4,5]:
            try: 
                if gamestates[gamestates['period_id']==id].iloc[-1]['type_name'] != "goal": 
                    neither_idx.append(gamestates[gamestates['period_id']==id].index[-1])
            except: break
    for idx in neither_idx:
        reward[idx] = [0,0,1]  

    dataset['Reward'] = reward
    return dataset


