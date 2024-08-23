import os, sys

target_dir = os.path.dirname(os.path.dirname(os.getcwd()))
# target_dir = os.path.join(home_dir, "K-league")
sys.path.insert(0, target_dir)

import json
import tqdm
tqdm.tqdm.pandas()
from typing import Any, Optional, cast

import pandas as pd  # type: ignore
from pandera.typing import DataFrame

class BeproLoader():
    def __init__(self):
        
        pass

    def competitions(self) -> DataFrame:
        """Return a dataframe with all available competitions and seasons.
            
        file : league.json & season.json

        """
        cols = [
            "season_id",
            "competition_id",
            "competition_name",
            "country_name",
            "season_name",
        ]

        league = self.read_json(path = "K-league/data/league.json")
        season = self.read_json(path = "K-league/data/season.json")

        competitions = []

        for i, row in enumerate(league["result"]):
            competition_id = row["id"]
            competition_name = row["name_en"]
            country_name = row["iso_country_code"]

            # 하나의 리그에도 여러 시즌 정보가 있을 수도 있음
            season_ids = row["season_ids"]
            for season_id in season_ids:
                season_name = next(filter(lambda x : x["id"] == season_id, season["result"]))["name"]
                competitions.append([season_id, competition_id, competition_name, country_name, season_name])
            
        return pd.DataFrame(competitions, columns=cols)

    def games(self, competition_id: int, season_id: int) -> DataFrame:
        """Return a dataframe with all available games in a season.

        file : info.json

        Parameters
        ----------
        competition_id : int
            The ID of the competition.
        season_id : int
            The ID of the season.
        """

        cols = [
            "match_id",
            "season_id",
            "competition_id",
            "round",
            "game_date",
            "home_team_id",
            "away_team_id",
            "home_score",
            "away_score",
            "venue",
        ]
        gamesdf = []
        match_ids = os.listdir(os.path.join(target_dir, "K-league/data/match"))

        for match_id in match_ids:
            match_info = self.read_json(path = f'K-league/data/match/{match_id}/info.json')["result"]

            # 원하는 대회 & 시즌이 아닌 경우
            if (season_id != match_info["season"]["id"]) or (competition_id != match_info["season"]["league_id"]):
                continue

            round = int(match_info["round"]["name"].split()[1])
            game_date = match_info["start_time"]

            home_team_id, away_team_id = match_info["home_team"]["id"], match_info["away_team"]["id"]
            home_score, away_score = match_info["detail_match_result"]["home_team_score"], match_info["detail_match_result"]["away_team_score"]

            venue = match_info["venue"]["display_name"]

            gamesdf.append([match_id, season_id, competition_id, round, game_date, 
                            home_team_id, away_team_id, home_score, away_score, venue])

        return pd.DataFrame(gamesdf, columns = cols)

    def _lineups(self, match_id: int) -> list:
        lineup = self.read_json(path = f'K-league/data/match/{match_id}/lineup.json')["result"]

        return lineup
    
    def teams(self, match_id: int) -> DataFrame:
        """Return a dataframe with both teams that participated in a game.

        file : team.json

        Parameters
        ----------
        match_id : int
            The ID of the game.
        """

        cols = ["team_id", "team_name_en", "team_name_ko"]

        # team.json 필요한 이유 : lineup정보는 team_id만 있고, team_name은 존재하지 않음
        team = pd.DataFrame(self.read_json(path = f'K-league/data/team.json')["result"])
        lineup = pd.DataFrame(self._lineups(match_id))

        obj = pd.merge(lineup, team, left_on = "team_id", right_on = "id", how = "left")
        obj.rename(columns={"name_en" : "team_name_en", "name" : "team_name_ko"}, inplace=True)

        return obj[cols]

    def players(self, match_id: int) -> DataFrame:
        """Return a dataframe with all players that participated in a game.

        file : lineup.json, player.json, player_stats.json

        Parameters
        ----------
        match_id : int
            The ID of the game.

        """
    
        cols = [
            "match_id",               # 파라미터
            "team_id",               # lineup
            "player_id",             # lineup
            "player_name_en",        # player(영어)
            "player_name_ko",        # lineup(한국어)
            "back_number",           # lineup(back_number)..콩글리쉬이긴 함...Jersey number지만 일단 회사따름
            "is_starting_lineup",    # lineup
            "position_name",         # lineup(position_name)
            "main_position",         # player(main_position)
            "play_time",             # player_stats(play_time), milliseconds
        ]

        lineup = pd.DataFrame(self._lineups(match_id))
        lineup["match_id"] = match_id
        
        # 팀별 player information추출 : 선수full이름, 선수full영문이름, 메인포지션, 생년월일등...
        teamA = self.read_json(path = f'K-league/data/player/team_{lineup["team_id"].unique()[0]}.json')["result"]
        teamB = self.read_json(path = f'K-league/data/player/team_{lineup["team_id"].unique()[1]}.json')["result"]
        team_df = pd.concat([pd.DataFrame(teamA), pd.DataFrame(teamB)]).reset_index()
        lineup = pd.merge(lineup, team_df, left_on = ["team_id", "player_id"], right_on = ["team_id", "id"], how="left", suffixes=("","_drop"))
        lineup["player_name_ko"] = lineup["player_last_name"] + lineup["player_name"]
        lineup["player_name_en"] = lineup["player_last_name_en"] + lineup["player_name_en"]
        
        # 선수별 stat정보 추출 : 선수 총 뛴 시간(추가로 사용하고 싶은 지표있으면 사용)
        player_stats = self.player_stats(match_id = match_id)
        lineup = pd.merge(lineup, player_stats, on = ["team_id", "player_id"], how="left", suffixes=("","_drop"))

        return lineup[cols]

    def events(self, match_id: int) -> DataFrame:
        """Return a dataframe with the event stream of a game.

        Parameters
        ----------
        match_id : int
            The ID of the game.

        """
        cols = [
            "match_id",
            "event_id",
            "event_period",
            "team_id",
            "player_id",
            "event_time",           # milliseconds
            "x",
            "y",
            "eventType",
            "subEventType",
            "cross",
            "outcome",
            "keyPass",
            "assist",
            "bodyPart",
            "relative_id",
            "relative_event_time",
            "relative_player_id",
            "relative_x",
            "relative_y",
            "ball_position_x",
            "ball_position_y",
            "event_types",
            "relative_event",
            "ball_position",
        ]

        events = pd.DataFrame(self.read_json(path = f'K-league/data/match/{match_id}/event_data.json')["result"])
        events.rename(columns = {"id" : "event_id"}, inplace=True)

        event_types_keys = ["eventType", "subEventType", "cross", "outcome", "keyPass", "assist", "bodyPart"]
        event_types = events["event_types"].apply(self.unpack_json, args=(event_types_keys,))

        relative_event_keys = ["id", "event_time", "player_id", "x", "y"]
        relative_event = events["relative_event"].apply(self.unpack_json, args=(relative_event_keys,))
        relative_event = relative_event.add_prefix('relative_')

        ball_position_keys = ["x", "y"]
        ball_position = events["ball_position"].apply(self.unpack_json, args=(ball_position_keys,))   
        ball_position = ball_position.add_prefix('ball_position_')
 
        return pd.concat([events, event_types, relative_event, ball_position], axis=1)[cols]

    def player_stats(self, match_id) -> DataFrame:
        player_stats = self.read_json(path = f'K-league/data/match/{match_id}/player_stats.json')

        data = []
        cols = ["team_id", "player_id", "play_time"]

        for team in player_stats["result"]:
            team_id = team['team_id']

            for player in team['players']:
                player_id = player['player_id']

                stats = player['stats']
                stats['team_id'] = team_id
                stats['player_id'] = player_id
                
                data.append(stats)

        return pd.DataFrame(data, columns = cols) 
    
    def read_json(self, path):
        # 절대 경로를 사용하여 파일을 로딩
        file_path = os.path.join(target_dir, path)
        with open(file_path, "r") as f:
            data = json.load(f)

        return data
    
    def unpack_json(self, json_object, json_keys) -> pd.DataFrame:
        """주어진 JSON 데이터를 처리하여 각 key를 개별 데이터로 확장합니다.

        Parameters
        ----------
        json_objects : JSON 객체
        json_keys : JSON 객체에서 추출할 key목록

        Returns:
            dict: 확장된 데이터
        """

        if isinstance(json_object, dict):
            unpacked_data = {key : None for key in json_keys}

            for key in json_keys:
                unpacked_data[key] = json_object.get(key, None)
        elif isinstance(json_object, list):                   # 여러가지 이벤트 유형이 존재하므로 통합시킴
            unpacked_data = {key: [] for key in json_keys}

            for obj in json_object:
                for key in json_keys:
                    unpacked_data[key].append(obj.get(key, None))
        else:
            unpacked_data = {key: None for key in json_keys}

        return pd.Series(unpacked_data)
