# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from tqdm import tqdm

# 1. 경기장 X, Y를 [100, 100]으로 조정
# socceraction.spadl.config 에서 field_length, field_width 조정 완료

# 2. 경기는 무조건 왼쪽에서 오른쪽으로 진행
def play_left_to_right(gamestates, home_team_id: int, away_team_id: int):

    field_length = 100
    field_width = 100

    """Perform all actions in a gamestate in the same playing direction.

    This changes the start and end location of each action in a gamestate,
    such that all actions are performed as if the team that performs the first
    action in the gamestate plays from left to right.

    Parameters
    ----------
    gamestates : GameStates
        The game states of a game.
    home_team_id : int
        The ID of the home team.

    Returns
    -------
    GameStates
        The game states with all actions performed left to right.

    See Also
    --------
    socceraction.vaep.features.play_left_to_right : For transforming actions.
    """
    away_idx = list(gamestates[gamestates['team_id'] == away_team_id].index)
    for index, idx in enumerate(away_idx):
        for col in ["start_x", "end_x"]:
            gamestates.loc[idx, col] = 100 - gamestates.loc[idx, col]
        for col in ["start_y", "end_y"]:
            gamestates.loc[idx, col] = 100 - gamestates.loc[idx, col]
    return gamestates


# 3. Goal sequence 삽입 -> Goal 단위(episode)로 학습되기 때문. 
def goal_sequence(dataset):
    goal = dataset[(dataset['type_name'].str.contains('shot'))&(dataset['result_name'] =='success')]
    goal[['start_x', 'end_x']] = 100
    goal[['start_y', 'end_y']] = 50
    goal['type_name'] = 'goal'
    goal['duration'] = 0
    # Own goal(자책골)
    owngoal = dataset[dataset['result_name'] =='owngoal']
    owngoal[['start_x', 'end_x']] = 100
    owngoal[['start_y', 'end_y']] = 50
    owngoal['type_name'] = 'owngoal'
    owngoal['duration'] = 0    

    dataset_with_goal_sequence = pd.concat([dataset, goal]).sort_values(['game_id','period_id','time_seconds'])
    dataset_with_goal_sequence = pd.concat([dataset_with_goal_sequence, owngoal]).sort_values(['game_id','period_id','time_seconds'])
    dataset_with_goal_sequence.reset_index(drop=True, inplace=True)
    return dataset_with_goal_sequence