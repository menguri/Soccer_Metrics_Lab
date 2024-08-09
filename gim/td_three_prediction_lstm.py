import csv
import tensorflow as tf
import os
import scipy.io as sio
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from nn.td_two_tower_lstm import TD_Prediction_TT_Embed
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from utils import handle_trace_length, get_together_training_batch, compromise_state_trace_length
from configuration import MODEL_TYPE, MAX_TRACE_LENGTH, FEATURE_NUMBER, BATCH_SIZE, GAMMA, H_SIZE, \
    model_train_continue, FEATURE_TYPE, ITERATE_NUM, learning_rate, SPORT, save_mother_dir, hidden_dim, lr, gamma, batch_size, memory_size, max_trace_length, output_dim, num_layers

LOG_DIR = str(os.getcwd()) + save_mother_dir + "/models/hybrid_sl_log_NN/Scale-three-cut_together_log_train_feature" + str(
    FEATURE_TYPE) + "_batch" + str(
    BATCH_SIZE) + "_iterate" + str(
    ITERATE_NUM) + "_lr" + str(
    learning_rate) + "_" + str(MODEL_TYPE) + "_MaxTL" + str(MAX_TRACE_LENGTH)
SAVED_NETWORK = str(os.getcwd()) + save_mother_dir + "/models/hybrid_sl_saved_NN/Scale-three-cut_together_saved_networks_feature" + str(
    FEATURE_TYPE) + "_batch" + str(
    BATCH_SIZE) + "_iterate" + str(
    ITERATE_NUM) + "_lr" + str(
    learning_rate) + "_" + str(MODEL_TYPE) + "_MaxTL" + str(MAX_TRACE_LENGTH)


DATA_STORE = "./datastore"
DIR_GAMES_ALL = os.listdir(DATA_STORE)
number_of_total_game = len(DIR_GAMES_ALL)


# Experiment tracking : Wandb
import wandb
wandb.init(project='Sarsa with LSTM')
wandb.run.name = 'td_three_prediction_1'
wandb.run.save()
args = {
    "learning_rate": lr,
}
wandb.config.update(args)


# Cost 저장
def write_game_average_csv(data_record):
    """
    write the cost of training
    :param data_record: the recorded cost dict
    """
    try:
        if os.path.exists(LOG_DIR + '/avg_cost_record.csv'):
            with open(LOG_DIR + '/avg_cost_record.csv', 'a') as csvfile:
                fieldnames = (data_record[0]).keys()
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                for record in data_record:
                    writer.writerow(record)
        else:
            with open(LOG_DIR + '/avg_cost_record.csv', 'w') as csvfile:
                fieldnames = (data_record[0]).keys()
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for record in data_record:
                    writer.writerow(record)
    except:
        if os.path.exists(LOG_DIR + '/avg_cost_record2.csv'):
            with open(LOG_DIR + '/avg_cost_record.csv', 'a') as csvfile:
                fieldnames = (data_record[0]).keys()
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                for record in data_record:
                    writer.writerow(record)
        else:
            with open(LOG_DIR + '/avg_cost_record2.csv', 'w') as csvfile:
                fieldnames = (data_record[0]).keys()
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for record in data_record:
                    writer.writerow(record)


def train_network(model):
    """
    training thr neural network game by game
    :param sess: session of tf
    :param model: nn model
    :return:
    """
    game_number = 0
    global_counter = 0
    converge_flag = False

    # Check if CUDA is available and use GPU if possible
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model.to(device)

    # instance of Pytorch Model
    writer = SummaryWriter(LOG_DIR)

    # loading network
    if model_train_continue:
        checkpoint = torch.load(SAVED_NETWORK)  # Load checkpoint
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        check_point_game_number = checkpoint['epoch']
        game_number_checkpoint = check_point_game_number % number_of_total_game
        game_number = check_point_game_number
        game_starting_point = 0
        print("Successfully loaded:", SAVED_NETWORK)
    else:
        print("Could not find old network weights")

    game_diff_record_all = []

    # 모든 게임 - 30 iteration
    while True:
        game_diff_record_dict = {}
        iteration_now = game_number / number_of_total_game + 1
        game_diff_record_dict.update({"Iteration": iteration_now})
        if converge_flag:
            break
        elif game_number >= number_of_total_game * ITERATE_NUM:
            break
        else:
            converge_flag = True
        # 게임 불러오기
        for dir_game in DIR_GAMES_ALL:

            if model_train_continue:
                if checkpoint and checkpoint.model_checkpoint_path:  # go the check point data
                    game_starting_point += 1
                    if game_number_checkpoint + 1 > game_starting_point:
                        continue

            v_diff_record = []
            game_number += 1
            game_cost_record = []
            game_files = os.listdir(DATA_STORE + "/" + dir_game)
            # 게임 안의 episode 별로 reward, input, trace가 저장되어 있다.
            for filename in game_files:
                if "rnn_reward" in filename:
                    reward_name = filename
                elif "rnn_input" in filename:
                    state_input_name = filename
                elif "trace" in filename:
                    state_trace_length_name = filename

            reward = sio.loadmat(DATA_STORE + "/" + dir_game + "/" + reward_name)
            reward = (reward['data'])[0]

            try:
                print(f"{dir_game} episode length : {len(reward)}")
            except:
                print("\n" + dir_game)
                raise ValueError("reward wrong")
            
            # episode length가 2도 안될 경우, s_t0 값만 있기에 학습 불가
            # 경기 시작하자마자 슛 -> 슛과 골 이벤트가 기록되어 있어 length=2
            if len(reward) != 2:
                continue

            # state_input 로드
            state_input = sio.loadmat(DATA_STORE + "/" + dir_game + "/" + state_input_name)
            state_input = (state_input['data'])[0]
            state_trace_length = sio.loadmat(DATA_STORE + "/" + dir_game + "/" + state_trace_length_name)
            state_trace_length = (state_trace_length['data'])[0]
            state_trace_length, state_input, reward = compromise_state_trace_length(state_trace_length, state_input,
                                                                                    reward, MAX_TRACE_LENGTH)

            reward_count = len(reward)
            print("reward number" + str(reward_count))
            print("=> load file" + str(dir_game) + " success")
            if len(state_input) != len(reward) or len(state_trace_length) != len(reward):
                raise Exception('state length does not equal to reward length')

            train_len = len(state_input)
            train_number = 0
            s_t0 = state_input[train_number]
            train_number += 1

            while True:
                # try:
                batch_return, train_number, s_tl, home_away_indicator = get_together_training_batch(s_t0,
                                                                                    state_input,
                                                                                    reward,
                                                                                    train_number,
                                                                                    train_len,
                                                                                    state_trace_length,
                                                                                    BATCH_SIZE)

                # get the batch variables
                s_t0_batch = [d[0] for d in batch_return]
                s_t1_batch = [d[1] for d in batch_return]
                r_t_batch = [d[2] for d in batch_return]
                trace_t0_batch = [d[3] for d in batch_return]
                trace_t1_batch = [d[4] for d in batch_return]
                y_batch = []

                # print(f"s_t0 : {s_t0_batch[-1]}")
                # print(f"s_t1 : {s_t1_batch[-1]}")
                # print(f"r_t_batch : {r_t_batch[-1]}")
                # print(f"trace_t0_batch : {trace_t0_batch}")
                # print(f"trace_t1_batch : {trace_t1_batch}")

                # Target 값 계산
                trace_t1_batch_tensor = torch.tensor(trace_t1_batch, dtype=torch.int32)
                s_t1_batch_tensor = torch.tensor(s_t1_batch, dtype=torch.float32)
                home_away_indicator_tensor = torch.tensor(home_away_indicator, dtype=torch.bool)
                # forward pass를 통해 출력 계산
                with torch.no_grad():
                    outputs_t1 = model.forward(s_t1_batch_tensor, trace_t1_batch_tensor, home_away_indicator_tensor[:, 1])
                # 홈/어웨이 출력 선택에 따라 readout_t1_batch 결정
                # readout_t1_batch = torch.where(home_away_indicator_tensor[:, 1], outputs_t1[:, 0], outputs_t1[:, 1])

                # 필요 시 numpy 배열로 변환 (TensorFlow의 sess.run()과 동일한 역할)
                readout_t1_batch = outputs_t1.numpy()

                # print(f"readout_t1_batch : {readout_t1_batch}")

                for i in range(0, len(batch_return)):
                    terminal = batch_return[i][5]
                    cut = batch_return[i][6]
                    # if terminal, only equals reward
                    if terminal or cut:
                        y_home = float((r_t_batch[i])[0])
                        y_away = float((r_t_batch[i])[1])
                        y_end = float((r_t_batch[i])[2])
                        y_batch.append([y_home, y_away, y_end])
                        break
                    else:
                        y_home = float((r_t_batch[i])[0]) + GAMMA * (readout_t1_batch[i]).tolist()[0]
                        y_away = float((r_t_batch[i])[1]) + GAMMA * (readout_t1_batch[i]).tolist()[1]
                        y_end = float((r_t_batch[i])[2]) + GAMMA * (readout_t1_batch[i]).tolist()[2]

                        wandb.log({"Home_prob": (readout_t1_batch[i]).tolist()[0]})
                        wandb.log({"Away_prob": (readout_t1_batch[i]).tolist()[1]})
                        wandb.log({"End_prob": (readout_t1_batch[i]).tolist()[2]})
                        # print(f"no terminal or cut : {[y_home, y_away, y_end]}")
                        y_batch.append([y_home, y_away, y_end])

                # y_batch를 텐서로 변환
                y_batch_tensor = torch.tensor(y_batch, dtype=torch.float32)
                # trace_t0_batch와 s_t0_batch를 텐서로 변환
                trace_t0_batch_tensor = torch.tensor(trace_t0_batch, dtype=torch.int32)
                s_t0_batch_tensor = torch.tensor(s_t0_batch, dtype=torch.float32)

                # s_t0 - forward | 손실 계산
                home_loss, away_loss, read_out = model.train_step(s_t0_batch_tensor, trace_t0_batch_tensor, home_away_indicator_tensor[:, 0], y_batch_tensor)
                
                # 출력 및 디버깅 정보
                diff = torch.mean(torch.abs(y_batch_tensor - read_out)).item()
                cost_out = torch.mean(torch.square(y_batch_tensor - read_out)).item()
                wandb.log({"td_diff": diff})
                wandb.log({"td_cost": cost_out})

                v_diff_record.append(diff)

                if cost_out > 0.0001:
                    converge_flag = False
                global_counter += 1
                game_cost_record.append(cost_out)
                # global_step 및 summary_train에 해당하는 값을 TensorBoard에 기록
                writer.add_scalar('Loss/train', cost_out, global_step=global_counter)
                writer.add_scalar('Diff/train', diff, global_step=global_counter)
                s_t0 = s_tl

                # print info
                if terminal or ((train_number - 1) / BATCH_SIZE) % 5 == 1:
                    print("TIMESTEP:", train_number, "Game:", game_number)
                    home_avg = sum(read_out[:, 0]) / len(read_out[:, 0])
                    away_avg = sum(read_out[:, 1]) / len(read_out[:, 1])
                    end_avg = sum(read_out[:, 2]) / len(read_out[:, 2])
                    print("home average:{0}, away average:{1}, end average:{2}".format(str(home_avg), str(away_avg),
                                                                                       str(end_avg)))
                    print("cost of the network is" + str(cost_out))

                if terminal:
                    # save progress after a game
                    save_path = os.path.join(SAVED_NETWORK, f"{SPORT}-game-{game_number}.pt")
                    # 모델의 상태 딕셔너리(가중치 등)를 저장
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': model.optimizer.state_dict(),
                        'game_number': game_number,
                        'global_step': global_counter
                    }, save_path)
                    v_diff_record_average = sum(v_diff_record) / len(v_diff_record)
                    game_diff_record_dict.update({dir_game: v_diff_record_average})
                    break

                    # break
            cost_per_game_average = sum(game_cost_record) / len(game_cost_record)
            write_game_average_csv([{"iteration": str(game_number / number_of_total_game + 1), "game": game_number,
                                     "cost_per_game_average": cost_per_game_average}])

        game_diff_record_all.append(game_diff_record_dict)


def train_start():

    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    if not os.path.exists(SAVED_NETWORK):
        os.makedirs(SAVED_NETWORK)


    if MODEL_TYPE == "two_tower":
        model = TD_Prediction_TT_Embed(FEATURE_NUMBER, hidden_dim, MAX_TRACE_LENGTH, learning_rate)
    else:
        raise ValueError("MODEL_TYPE error")
    train_network(model)


if __name__ == '__main__':
    train_start()
