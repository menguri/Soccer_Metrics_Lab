"""Kleague event stream data to SPADL converter."""

from typing import Any, cast

import numpy as np
import pandas as pd  # type: ignore
from pandera.typing import DataFrame

from . import config as spadlconfig
from .base import (
    _add_dribbles,
    _fix_clearances,
    # _fix_direction_of_play,
    min_dribble_length,
)
from .schema import SPADLSchema

HEIGHT_POST = 2.5
TOUCH_LINE_LENGTH = 105
GOAL_LINE_LENGTH = 68

LEFT_POST = 0.449 * GOAL_LINE_LENGTH # convert ratio to meter
RIGHT_POST = 0.551 * GOAL_LINE_LENGTH # convert ratio to meter
CENTER_POST = (LEFT_POST+RIGHT_POST) / 2

Eighteen_YARD = 16.4592 # 18yard = 16.4592meter

def convert_to_actions(events: pd.DataFrame, home_team_id: int) -> DataFrame[SPADLSchema]:
    """
    Convert K-league events to SPADL actions.
    """

    # 승부차기는 API에 없음
    period_dict = {"FIRST_HALF" : 1, "SECOND_HALF" : 2, "EXTRA_FIRST_HALF" : 3, "EXTRA_SECOND_HALF" : 4} 
    events["period_id"] = events["event_period"].map(period_dict)

    # event_id는 경기별로 이벤트 시간에 따라 부여된 id가 아님
    events = events.sort_values(["match_id", "period_id", "event_time"]).reset_index(drop=True)

    # 빈 리스트가 아닌 eventType만 남기기
    events = events[events['eventType'].apply(len) > 0].reset_index(drop=True)

    # K-league경기장 규격에 맞춘 (x, y)좌표를 meter로 변환
    events = _convert_locations(events)

    # 기록되어있지 않는 슛의 끝위치를 대체함
    events = create_shot_coordinates(events)
 
    # 홈팀과 원정팀의 play방향을 고정
    # 홈팀은 아래->위쪽에서 플레이하고 원정팀은 위쪽->아래쪽로 플레이를 수행함
    events = _fix_direction_of_play(events, home_team_id)

    # 공격이벤트과 수비이벤트가 동시에 발생한 경우 split함
    events = insert_defensive_actions(events, defensive_action="Interception")
    events = insert_defensive_actions(events, defensive_action="Tackle")

    # K-league데이터셋 형태의 aciton을 SPALD형태로 변환
    events["type_name"] = events[["eventType", "subEventType", "cross"]].apply(_get_type_name, axis=1)
    events = _fix_recoveries(events)

    # 각 액션에 따리 현재 or 미래의 정보(위치 or 이벤트정보)를 활용하므로 사전에 저장
    # ex) Carry, Dribble, Clearance
    events = extract_previous_next_xy(events)

    # 각 액션에 따라 액션id, 터치부위, 결과, 끝 위치를 정의함
    events[["type_id", "bodypart_id", "result_id", "end_x", "end_y"]] = events.apply(_parse_event, axis=1, result_type="expand")
    events = adjust_dribble_end_location(events)
    events = remove_non_actions(events)  # remove all non-actions left
    events = adjust_interception_results(events)
    
    actions = pd.DataFrame()

    actions["game_id"] = events.match_id.astype(int)
    actions["original_event_id"] = events.event_id.astype(object)
    actions["period_id"] = events.period_id.astype(int)
    actions["time_seconds"] = events["event_time"] * 0.001 # convert milliseconds to seconds
    actions["team_id"] = events.team_id.astype(int)
    actions["player_id"] = events.player_id.astype(int)
    
    # K-leage경기장 형태를 SPADL형태로 변환 : 68x105 -> 105x68로 변환
    actions["start_x"] = events.y
    actions["start_y"] = GOAL_LINE_LENGTH - events.x
    actions["end_x"] = events.end_y
    actions["end_y"] = GOAL_LINE_LENGTH - events.end_x

    actions["type_id"] = events.type_id.astype(int)
    actions["bodypart_id"] = events.bodypart_id.astype(int)
    actions["result_id"] = events.result_id.astype(int)

    actions["action_id"] = range(len(actions))
    actions = _add_dribbles(actions)

    return cast(DataFrame[SPADLSchema], actions)

def _parse_event(event : pd.Series) -> tuple[int, int, float, float]:
    # 22 possible values : pass, cross, throw-in, 
    # crossed free kick, short free kick, crossed corner, short corner, 
    # take-on, foul, tackle, interception, 
    # shot, penalty shot, free kick shot, 
    # keeper save, keeper claim, keeper punch, keeper pick-up, 
    # clearance, bad touch, dribble and goal kick.
    events = {
        "pass": _parse_pass_event,
        "cross": _parse_pass_event,
        "throw_in": _parse_pass_event,
        "freekick_crossed": _parse_pass_event,
        "freekick_short": _parse_pass_event,
        "corner_crossed": _parse_pass_event,
        "corner_short": _parse_pass_event,

        "take_on": _parse_take_on_event,

        "foul": _parse_foul_event,

        "tackle" : _parse_tackle_event,

        "interception": _parse_interception_event,

        "shot": _parse_shot_event,
        "shot_penalty": _parse_shot_event,
        "shot_freekick": _parse_shot_event,

        "keeper_save" : _parse_goalkeeper_event,
        "keeper_claim" : _parse_goalkeeper_event,
        "keeper_punch" : _parse_goalkeeper_event,
        "keeper_pick_up" : _parse_goalkeeper_event,

        "clearance" : _parse_clearance_event,
        "bad_touch" : _parse_bad_touch_event,
        "dribble" : _parse_dribble_event,

        "goalkick" : _parse_pass_event,
    }

    parser = events.get(event["type_name"], _parse_event_as_non_action)
    bodypart, result, end_x, end_y = parser(event)

    actiontype = spadlconfig.actiontypes.index(event["type_name"])
    bodypart = spadlconfig.bodyparts.index(bodypart)
    result = spadlconfig.results.index(result)
    
    return actiontype, bodypart, result, end_x, end_y

def _get_type_name(args: tuple[list, list, list]) -> str:
    eventType, sub_event_type, cross = args

    if "Pass" in eventType:
        if True in cross:
            if "Freekick" in sub_event_type: 
                a = "freekick_crossed"
            elif "Corner" in sub_event_type:
                a = "corner_crossed"
            else:
                a = "cross"
        else:
            if "Freekick" in sub_event_type: 
                a = "freekick_short"
            elif "Corner" in sub_event_type:
                a = "corner_short"
            elif "Throw-In" in sub_event_type:
                a = "throw_in"
            elif "Goal Kick" in sub_event_type:
                a = "goalkick"
            else:
                a = "pass"
    elif "Shot" in eventType or "Own Goal" in eventType:
        if "Freekick" in sub_event_type:
            a = "shot_freekick"
        elif "Penalty Kick" in sub_event_type:
            a = "shot_penalty"
        else:
            a = "shot"
    elif "Take-On" in eventType:
        a = "take_on"
    elif "Carry" in eventType:
        a = "dribble"
    elif "Save" in eventType: # 골키퍼 액션 : Save, Aerial Clearnce, Defensive Line Support Succeeded     
        if "Catch" in sub_event_type:
            a = "keeper_save"
        elif "Parry" in sub_event_type:
            a = "keeper_punch"
        else:
            a = "non_action" # Save는 Catch과 Parry만 존재하고 예외경우는 존재하지는 않음
    elif "Aerial Clearance" in eventType:
        a = "keeper_claim"
    elif "Defensive Line Support" in eventType: # 무엇으로 정의하기가 어려운 액션
        a = "non_action"
    elif "Foul" in eventType:
        a = "foul"
    elif "Tackle" in eventType:
        a = "tackle"
    elif "Interception" in eventType or "Intervention" in eventType or "Block" in eventType:
        a = "interception"
    elif "Clearance" in eventType:
        a = "clearance"
    elif "Error" in eventType:
        a = "bad_touch"
    else:
        a = "non_action"

    return a

# bodypart는 여러 터치부위가 동시에 등장하는 경우는 없음
def _get_bodypart_name(args : tuple[list]) -> str:
    bodyPart = args

    if "Hands" in bodyPart:
        b = "other"
    elif "Head" in bodyPart:
        b = "head"
    elif "Left Foot" in bodyPart:
        b = "foot_left"
    elif "Right Foot" in bodyPart:
        b = "foot_right"
    elif "Lower Body" in bodyPart or "Upper Body" in bodyPart or "Other" in bodyPart:
        b = "other"
    else:
        b = None

    return b

def _fix_direction_of_play(df_events: pd.DataFrame, home_team_id: int) -> pd.DataFrame:
    away_idx = (df_events.team_id != home_team_id).values
    for col in ["x", "relative_x","end_x"]:
        df_events.loc[away_idx, col] = GOAL_LINE_LENGTH - df_events[away_idx][col].values
    for col in ["y", "relative_y", "end_y"]: 
        df_events.loc[away_idx, col] = TOUCH_LINE_LENGTH - df_events[away_idx][col].values

    return df_events

# recovery액션도 드리블로 변환하여 사용함
def _fix_recoveries(df_events: pd.DataFrame) -> pd.DataFrame:
    """Convert ball recovery events to dribbles.
    """
    df_events_next = df_events.shift(-1)
    selector_recovery = df_events["eventType"].apply(lambda x : "Recovery" in x)
    non_action = (df_events["type_name"] == "non_action") # 패스, 슛과 동시에 부여된 recovery는 해당 액션으로 매핑

    same_x = abs(df_events["x"] - df_events_next["x"]) < min_dribble_length
    same_y = abs(df_events["y"] - df_events_next["y"]) < min_dribble_length
    same_loc = same_x & same_y

    df_events.loc[selector_recovery & non_action & ~same_loc, "type_name"] = "dribble"
    df_events.loc[selector_recovery & non_action &  same_loc, "type_name"] = "non_action"
    
    return df_events

# 드리블(take_on, dribble)이 성공했을 때와 실패했을 때의 끝 위치를 업데이트하는 함수
# 실패한 경우: 드리블을 방어한 수비 이벤트의 위치를 사용(take_on, dribble parse함수에서 처리)
# 성공한 경우: 다음 이벤트가 tackle 또는 interception이 아닌 이벤트의 위치를 사용
def adjust_dribble_end_location(df_events: pd.DataFrame) -> pd.DataFrame:
    defensive_events = ['tackle', 'interception']
    offensive_events = ['take_on', 'dribble']

    # 성공한 드리블(take_on, dribble)
    successful_offensive_indices = df_events[
        (df_events['type_name'].isin(offensive_events)) & (df_events['result_id'] == spadlconfig.results.index("success"))
    ].index.tolist()
    

    # 성공한 드리블(take_on, dribble)이후 tackle & interception가 아닌 첫번째 이벤트.
    # idx+1 ~ idx+5번째 이벤트를 확인
    for idx in successful_offensive_indices:
        # 현재 인덱스 이후 최대 3개의 이벤트를 확인
        max_index = min(idx + 4, len(df_events))
        for next_index in range(idx+1, max_index):
          
            if (
                (df_events.at[idx, "period_id"] == df_events.at[next_index, "period_id"]) &
                (df_events.at[next_index, 'type_name'] not in defensive_events)
            ):
                # 해당 이벤트의 위치로 드리블 종료 위치 업데이트
                df_events.at[idx, 'end_x'] = df_events.at[next_index, 'x']
                df_events.at[idx, 'end_y'] = df_events.at[next_index, 'y']
                break  # 첫 번째 유효한 이벤트를 찾으면 반복 종료

    return df_events

#Interception 이벤트 이후 Hit, Duel 등의 경합과정이 있을 경우 이벤트 결과가 제대로 반영되지 않는 문제를 해결하는 함수
def adjust_interception_results(df_events: pd.DataFrame) -> pd.DataFrame:
    interception_indices = df_events[df_events['type_name'] == "interception"].index
    interception_indices = interception_indices[interception_indices+1 < len(df_events)] # 다음이벤트가 존재하지 않는 경우
    
    for idx in interception_indices:
        unsuccess_interception_condition = (
            (df_events.at[idx, "period_id"] == df_events.at[idx+1, "period_id"]) &
            (df_events.at[idx, "team_id"] != df_events.at[idx+1, "team_id"])
        )

        if unsuccess_interception_condition:
            df_events.at[idx, "result_id"] = spadlconfig.results.index("fail")
        else:
            df_events.at[idx, "result_id"] = spadlconfig.results.index("success")
    return df_events

# 공격 이벤트와 수비 이벤트가 함께 존재하는지 확인하는 함수
def insert_defensive_actions(df_events: pd.DataFrame, defensive_action : str) -> pd.DataFrame:
    """Insert defensive actions before offensive actions

    """

    # 공격 이벤트 목록
    attack_events = ["Pass", "Shot", "Take-On", "Carry"]

    def is_defensive_action_attack(eventTypes : list) -> bool:
        has_attack = any(eventType in attack_events for eventType in eventTypes)
        has_defense = defensive_action in eventTypes

        return has_attack and has_defense
    
    df_events_defense = df_events[df_events["eventType"].apply(is_defensive_action_attack)].copy()
    
    if not df_events_defense.empty:
        df_events_defense["eventType"] = [[defensive_action] for _ in range(len(df_events_defense))]
        df_events_defense["outcome"] = [["Successful"] for _ in range(len(df_events_defense))] # 수비액션 후에 공격액션을 수행하므로 수비액션은 항상 성공
        df_events_defense["cross"] = [[None] for _ in range(len(df_events_defense))]
        df_events_defense["bodyPart"] = [[None] for _ in range(len(df_events_defense))]

        df_events = pd.concat([df_events_defense, df_events], ignore_index=True)
        df_events = df_events.sort_values(["period_id", "event_time"], kind="mergesort")
        df_events = df_events.reset_index(drop=True)

    return df_events

def remove_non_actions(df_events: pd.DataFrame) -> pd.DataFrame:
    """Remove the remaining non_actions from the action dataframe.

    """
    df_events = df_events[df_events["type_id"] != spadlconfig.actiontypes.index("non_action")]
    # remove remaining ball out of field, whistle and goalkeeper from line
    df_events = df_events.reset_index(drop=True)
    return df_events


def _convert_locations(df_events: pd.DataFrame) -> pd.DataFrame:
    """Convert StatsBomb locations to spadl coordinates.
    
    K-league 경기장 규격 특징
    1. 68x105
    2. (0,0)은 bottom-left이고 (1,1)은 upper-right이다.
    3. 경기 half랑 상관없이 이벤트는 항상 Goal line(y=0)에서 시작한다.
    4. x *= 68(GOAL_LINE_LENGTH), y *= 105(TOUCH_LINE_LENGTH)
    """

    df_events[["x", "relative_x"]] = np.clip(df_events[["x", "relative_x"]] * GOAL_LINE_LENGTH, 0, GOAL_LINE_LENGTH)
    df_events[["y", "relative_y"]] = np.clip(df_events[["y", "relative_y"]] * TOUCH_LINE_LENGTH, 0, TOUCH_LINE_LENGTH)

    df_events["ball_position_x"] = LEFT_POST + ((RIGHT_POST - LEFT_POST) * df_events["ball_position_x"])
    df_events["ball_position_y"] = df_events["ball_position_y"] * HEIGHT_POST

    return df_events


def create_shot_coordinates(df_events: pd.DataFrame) -> pd.DataFrame:
    shot = df_events["eventType"].apply(lambda x : "Shot" in x)
    owngoal = df_events["eventType"].apply(lambda x : "Own Goal" in x)

    # 왼쪽 측면에서 슛팅
    out_left_idx = (
        df_events["x"] < (LEFT_POST - Eighteen_YARD)
    )
    df_events.loc[shot & out_left_idx, "end_x"] = LEFT_POST - Eighteen_YARD
    df_events.loc[shot & out_left_idx, "end_y"] = TOUCH_LINE_LENGTH

    # 오른쪽 측면에서 슛팅
    out_right_idx = (
        df_events["x"] > (RIGHT_POST + Eighteen_YARD)
    )
    df_events.loc[shot & out_right_idx, "end_x"] = RIGHT_POST + Eighteen_YARD
    df_events.loc[shot & out_right_idx, "end_y"] = TOUCH_LINE_LENGTH

    out_height_idx = (
        (df_events["x"] >= (LEFT_POST - Eighteen_YARD)) &
        (df_events["x"] <= (RIGHT_POST + Eighteen_YARD))
    )
    df_events.loc[shot & out_height_idx, "end_x"] = CENTER_POST
    df_events.loc[shot & out_height_idx, "end_y"] = TOUCH_LINE_LENGTH

    # 자책골
    df_events.loc[owngoal, "end_x"] = CENTER_POST
    df_events.loc[owngoal, "end_y"] = 0

    # 슛팅의 끝 위치가 기록되어 있는 경우
    target_shot_idx = (
        df_events["ball_position_x"].notna() &
        df_events["ball_position_y"].notna()
    )
    df_events.loc[shot & target_shot_idx, "end_x"] = df_events.loc[shot & target_shot_idx, "ball_position_x"]
    df_events.loc[shot & target_shot_idx, "end_y"] = TOUCH_LINE_LENGTH # 공의 높이 정보인 ball_position_y는 사용하지 않음

    # blocking당한 액션은 다음 액션인 블로킹이벤트의 시작위치
    # 주의사항 : blocking한 액션은 수비팀의 액션이기 때문에 y=0주위로 기록이 되어있음
    blocked_idx = ( shot &
                   df_events["outcome"].apply(lambda x: "Blocked" in x)
    )
    next_blocked_idx = df_events.index[blocked_idx] + 1
    df_events.loc[blocked_idx, "end_x"] = GOAL_LINE_LENGTH - df_events.loc[next_blocked_idx, "x"].values
    df_events.loc[blocked_idx, "end_y"] = TOUCH_LINE_LENGTH - df_events.loc[next_blocked_idx, "y"].values

    return df_events

def extract_previous_next_xy(df_events: pd.DataFrame) -> pd.DataFrame:
    # 그룹별로 이전과 이후의 x, y 값 이동
    # df_actions["prev_relative_x"] = df_actions.groupby("period_id")["relative_x"].shift(1)
    # df_actions["prev_relative_y"] = df_actions.groupby("period_id")["relative_y"].shift(1)

    # df_actions["prev_ball_position_x"] = df_actions.groupby("period_id")["ball_position_x"].shift(1)
    # df_actions["prev_ball_position_y"] = df_actions.groupby("period_id")["ball_position_y"].shift(1)

    df_events["next_x"] = df_events.groupby("period_id")["x"].shift(-1)
    df_events["next_y"] = df_events.groupby("period_id")["y"].shift(-1)
    df_events["next_eventType"] = df_events.groupby("period_id")["eventType"].shift(-1)
    df_events["next_teamId"] = df_events.groupby("period_id")["team_id"].shift(-1)

    return df_events

def _parse_event_as_non_action(event):
    bodypart = "foot"
    result = "success"
    end_x, end_y = 0, 0
    return bodypart, result, end_x, end_y

def _parse_pass_event(event):
    # bodypart정의
    if "Aerial" in event["subEventType"]:
        bodypart = "head" 
    elif "Throw-In" in event["subEventType"]:
        bodypart = "other"
    else:
        bodypart = _get_bodypart_name(event["bodyPart"])
        bodypart = bodypart if pd.notna(bodypart) else "foot"

    # result정의
    pass_outcome = [event_type["outcome"] for event_type in event['event_types'] if event_type.get("eventType") == "Pass"][0]
    next_eventType =  event["next_eventType"] if isinstance(event["next_eventType"], list) else [None]
    
    if pass_outcome == "Successful":
        result = "success"
    elif pass_outcome == "Unsuccessful":
        if "Offside" in next_eventType:
            result = "offside"
        else:
            result = "fail"
    else:
        result = None

    # 끝 위치 정의
    end_x, end_y = event[["relative_x", "relative_y"]]
    if pd.isna(end_x) or pd.isna(end_y):
        end_x, end_y = event[["next_x", "next_y"]] # 일부 패스는 결측기가 존재함
    
    return bodypart, result, end_x, end_y

def _parse_take_on_event(event):
    bodypart = _get_bodypart_name(event["bodyPart"])
    bodypart = bodypart if pd.notna(bodypart) else "foot"

    result = "success"

    # result정의
    take_on_outcome = [event_type["outcome"] for event_type in event['event_types'] if event_type.get("eventType") == "Take-On"][0]
    if take_on_outcome  == "Successful":
        result = "success"
    elif take_on_outcome  == "Unsuccessful":
        result = "fail"
    else:
        result = None

    end_x, end_y = event[["next_x", "next_y"]] # 추후에 adjust_dribble_end_location함수에서 성공한 take_on은 끝위치를 조정함
    if pd.isna(end_x) or pd.isna(end_y):
        end_x, end_y = event[["x", "y"]] # take_on이후 경기가 종료되는 경우

    return bodypart, result, end_x, end_y

def _parse_foul_event(event):
    if "Handball Foul" in event["subEventType"]:
        bodypart = "other" 
    else:
        bodypart = _get_bodypart_name(event["bodyPart"])
        bodypart = bodypart if pd.notna(bodypart) else "foot"

    if "Red Card" in event["subEventType"]:
        result = "red_card"
    elif "Yellow Card" in event["subEventType"]:
        result = "yellow_card"
    else:
        result = "fail"

    end_x, end_y = event[["x", "y"]]
    
    return bodypart, result, end_x, end_y

def _parse_tackle_event(event):
    if "Aerial" in event["subEventType"]:
        bodypart = "other" 
    else:
        bodypart = _get_bodypart_name(event["bodyPart"])
        bodypart = bodypart if pd.notna(bodypart) else "foot"

    tackle_outcome = [event_type["outcome"] for event_type in event['event_types'] if event_type.get("eventType") == "Tackle"][0]
    if tackle_outcome  == "Successful":
        result = "success"
    elif tackle_outcome  == "Unsuccessful":
        result = "fail"
    else:
        result = None

    end_x, end_y = event[["x", "y"]]
    
    return bodypart, result, end_x, end_y

def _parse_interception_event(event):
    if "Aerial" in event["subEventType"]:
        bodypart = "other" 
    else:
        bodypart = _get_bodypart_name(event["bodyPart"])
        bodypart = bodypart if pd.notna(bodypart) else "foot"

    result = "success" if event["team_id"] == event["next_teamId"] else "fail" # 추후에 adjust_interception_results함수에서 결과를 조정함

    end_x, end_y = event[["x", "y"]]
    
    return bodypart, result, end_x, end_y

def _parse_shot_event(event):
    bodypart = _get_bodypart_name(event["bodyPart"])
    bodypart = bodypart if pd.notna(bodypart) else "foot"

    # result정의
    if "Own Goal" in event["eventType"]:
        result = "owngoal"
    else:
        shot_outcome = [event_type["outcome"] for event_type in event['event_types'] if event_type.get("eventType") == "Shot"][0]
        if shot_outcome == "Goal":
            result = "success"
        else:
            result = "fail"

    end_x, end_y = event["end_x"], event["end_y"]

    return bodypart, result, end_x, end_y

def _parse_goalkeeper_event(event):
    bodypart = _get_bodypart_name(event["bodyPart"])
    bodypart = bodypart if pd.notna(bodypart) else "other"

    if "Save" in event["eventType"]:
        result = "success"
    elif "Aerial Clearance" in event["eventType"]:
        Aerial_Clearance_outcome = [event_type["outcome"] for event_type in event['event_types'] if event_type.get("eventType") == "Aerial Clearance"][0]

        if Aerial_Clearance_outcome  == "Successful":
            result = "success"
        elif Aerial_Clearance_outcome  == "Unsuccessful":
            result = "fail"
        else:
            result = None
    else:
        result = None

    end_x, end_y = event[["x", "y"]]

    return bodypart, result, end_x, end_y

def _parse_clearance_event(event):
    if "Aerial" in event["subEventType"]:
        bodypart = "head" 
    else:
        bodypart = _get_bodypart_name(event["bodyPart"])
        bodypart = bodypart if pd.notna(bodypart) else "foot"

    result = "success"

    end_x, end_y = event[["next_x", "next_y"]]  
    if pd.isna(end_x) or pd.isna(end_y):
        end_x, end_y = event[["x", "y"]] # clearance이후 경기가 종료되는 경우

    return bodypart, result, end_x, end_y

def _parse_bad_touch_event(event):
    bodypart = _get_bodypart_name(event["bodyPart"])
    bodypart = bodypart if pd.notna(bodypart) else "foot"

    result = "fail"
    end_x, end_y = event[["x", "y"]]
    
    return bodypart, result, end_x, end_y

def _parse_dribble_event(event):
    bodypart = _get_bodypart_name(event["bodyPart"])
    bodypart = bodypart if pd.notna(bodypart) else "foot"

    result = "success"

    end_x, end_y = event[["next_x", "next_y"]] # 추후에 adjust_dribble_end_location함수에서 성공한 take_on은 끝위치를 조정함
    if pd.isna(end_x) or pd.isna(end_y):
        end_x, end_y = event[["x", "y"]]

    return bodypart, result, end_x, end_y
