import numpy as np
from configuration import FEATURE_NUMBER


def handle_trace_length(state_trace_length):
    """
    transform format of trace length
    :return:
    """
    trace_length_record = []
    for length in state_trace_length:
        for sub_length in range(0, int(length)):
            trace_length_record.append(sub_length + 1)

    return trace_length_record


def get_together_training_batch(s_t0, state_input, reward, train_number, train_len, state_trace_length, BATCH_SIZE):
    """
    we generate the training batch, your can write your own method.
    in our dataset, 1 means home score, -1 means away score, we transfer it to one-hot representation:
    reward  = [If_home_score, If_away_score, If_NeitherTeam_score]
    :return:
    batch_return is [s,s',r,s_play_length,s'_play_length, if_game_end, if_score_in_the_last_time_step]
    train_number is the current where we stop training
    s_t0 is the s for the next batch
    """
    batch_return = []
    home_away_indicator = []
    current_batch_length = 0
    while current_batch_length < BATCH_SIZE:
        s_t1 = state_input[train_number]
        # if len(s_t1) < 10 or len(s_t0) < 10:
        #     raise ValueError("wrong length of s")
        #     train_number += 1
        #     continue
        s_length_t1 = state_trace_length[train_number]
        s_length_t0 = state_trace_length[train_number - 1]

        # home, away indicator | input 11번째 feature로 team(home:1.009xxx, away:-0.990xxx)이 들어가 있음.
        s_t0_indicator = 0
        s_t1_indicator = 0
        if s_t0[0][10] > 1:
            s_t0_indicator = 1
        if s_t1[0][10] > 1:
            s_t1_indicator = 1
        home_away_indicator.append([s_t0_indicator, s_t1_indicator])

        if s_length_t1 > 10:  # if trace length is too long
            s_length_t1 = 10
        if s_length_t0 > 10:  # if trace length is too long
            s_length_t0 = 10
        try:
            # reward는 [0]으로 한번 더 들어가줘야 함.
            # t1, t0의 reward를 추출
            s_reward_t1 = reward[train_number][0]
            s_reward_t0 = reward[train_number - 1][0]
        except IndexError:
            raise IndexError("s_reward wrong with index")
        
        print(f"train_number + 1 : {train_number + 1}")
        print(f"train_len : {train_len}")
        if train_number + 1 == train_len:
            print("여기 아니야?")
            # t1, t0의 index를 가져옴
            trace_length_index_t1 = s_length_t1 - 1
            trace_length_index_t0 = s_length_t0 - 1
            # t1, t0의 reward를 nparray 형태로 추출
            r_t0 = np.asarray([s_reward_t0[trace_length_index_t0]])
            r_t1 = np.asarray([s_reward_t1[trace_length_index_t1]])

            train_number += 1
            if r_t0 == [float(0)]:
                r_t0_combine = [float(0), float(0), float(0)]
                batch_return.append((s_t0, s_t1, r_t0_combine, s_length_t0, s_length_t1, 0, 0))

                if r_t1 == float(0):
                    r_t1_combine = [float(0), float(0), float(1)]
                elif r_t1 == float(-1):
                    r_t1_combine = [float(0), float(1), float(1)]
                elif r_t1 == float(1):
                    r_t1_combine = [float(1), float(0), float(1)]
                else:
                    raise ValueError("incorrect r_t1")

                # home, away indicator | input 11번째 feature로 team(home:1.009xxx, away:-0.990xxx)이 들어가 있음.
                s_t1_indicator = 0
                if s_t1[0][10] > 1:
                    s_t1_indicator = 1
                home_away_indicator.append([s_t1_indicator, s_t1_indicator])
                batch_return.append((s_t1, s_t1, r_t1_combine, s_length_t1, s_length_t1, 1, 0))

            elif r_t0 == [float(-1)]:
                r_t0_combine = [float(0), float(1), float(0)]
                batch_return.append((s_t0, s_t1, r_t0_combine, s_length_t0, s_length_t1, 0, 0))

                if r_t1 == float(0):
                    r_t1_combine = [float(0), float(0), float(1)]
                elif r_t1 == float(-1):
                    r_t1_combine = [float(0), float(1), float(1)]
                elif r_t1 == float(1):
                    r_t1_combine = [float(1), float(0), float(1)]
                else:
                    raise ValueError("incorrect r_t1")

                # home, away indicator | input 11번째 feature로 team(home:1.009xxx, away:-0.990xxx)이 들어가 있음.
                s_t1_indicator = 0
                if s_t1[0][10] > 1:
                    s_t1_indicator = 1
                home_away_indicator.append([s_t1_indicator, s_t1_indicator])
                batch_return.append((s_t1, s_t1, r_t1_combine, s_length_t1, s_length_t1, 1, 0))

            elif r_t0 == [float(1)]:
                r_t0_combine = [float(1), float(0), float(0)]
                batch_return.append((s_t0, s_t1, r_t0_combine, s_length_t0, s_length_t1, 0, 0))

                if r_t1 == float(0):
                    r_t1_combine = [float(0), float(0), float(1)]
                elif r_t1 == float(-1):
                    r_t1_combine = [float(0), float(1), float(1)]
                elif r_t1 == float(1):
                    r_t1_combine = [float(1), float(0), float(1)]
                else:
                    raise ValueError("incorrect r_t1")

                # home, away indicator | input 11번째 feature로 team(home:1.009xxx, away:-0.990xxx)이 들어가 있음.
                s_t1_indicator = 0
                if s_t1[0][10] > 1:
                    s_t1_indicator = 1
                home_away_indicator.append([s_t1_indicator, s_t1_indicator])
                batch_return.append((s_t1, s_t1, r_t1_combine, s_length_t1, s_length_t1, 1, 0))
            else:
                raise ValueError("r_t0 wrong value")

            s_t0 = s_t1
            break
        
        train_number += 1
        trace_length_index_t0 = s_length_t0 - 1
        r_t0 = np.asarray([s_reward_t0[trace_length_index_t0]])
        if r_t0 != [float(0)]:
            # print r_t0
            if r_t0 == [float(-1)]:
                r_t0_combine = [float(0), float(1), float(0)]
                batch_return.append((s_t0, s_t1, r_t0_combine, s_length_t0, s_length_t1, 0, 1))
            elif r_t0 == [float(1)]:
                r_t0_combine = [float(1), float(0), float(0)]
                batch_return.append((s_t0, s_t1, r_t0_combine, s_length_t0, s_length_t1, 0, 1))
            else:
                raise ValueError("r_t0 wrong value")
            s_t0 = s_t1
            break
        r_t0_combine = [float(0), float(0), float(0)]

        batch_return.append((s_t0, s_t1, r_t0_combine, s_length_t0, s_length_t1, 0, 0))
        current_batch_length += 1
        s_t0 = s_t1
    
    return batch_return, train_number, s_t0, np.array(home_away_indicator)


def padding_hybrid_feature_input(hybrid_feature_input):
    """
    padding the empty state features with 0 (states won't be traced by Dynamic LSTM)
    :param hybrid_feature_input: the lists of features state to be padding
    :return:
    """
    current_list_length = len(hybrid_feature_input)
    padding_list_length = 10 - current_list_length
    for i in range(0, padding_list_length):
        hybrid_feature_input.append(np.asarray([float(0)] * FEATURE_NUMBER))
    return np.asarray(hybrid_feature_input)


def padding_hybrid_reward(hybrid_reward):
    """
    padding the empty state rewards with 0 (rewards won't be traced by Dynamic LSTM)
    :param hybrid_reward: the lists of rewards to be padding
    :return:
    """
    current_list_length = len(hybrid_reward)
    padding_list_length = 10 - current_list_length
    for i in range(0, padding_list_length):
        hybrid_reward.append(0)
    return np.asarray([hybrid_reward])


def compromise_state_trace_length(state_trace_length, state_input, reward, MAX_TRACE_LENGTH):
    """
    padding the features and rewards with 0, in order to get a proper format for LSTM
    :param state_trace_length: list of trace length
    :param state_input: list of state
    :param reward: list of rewards
    """
    state_trace_length_output = []
    for index in range(0, len(state_trace_length)):
        tl = state_trace_length[index]
        if tl > MAX_TRACE_LENGTH:
            state_input_change_list = []
            state_input_org = state_input[index]
            reward_change_list = []
            reward_org = reward[index][0]
            for i in range(0, MAX_TRACE_LENGTH):
                state_input_change_list.append(state_input_org[tl - MAX_TRACE_LENGTH + i])
                # temp = reward_org[tl - MAX_TRACE_LENGTH + i]
                # if temp != 0:
                #     print 'find miss reward'
                reward_change_list.append(reward_org[tl - MAX_TRACE_LENGTH + i])

            state_input_update = padding_hybrid_feature_input(state_input_change_list)
            state_input[index] = state_input_update
            reward_update = padding_hybrid_reward(reward_change_list)
            reward[index] = reward_update

            tl = MAX_TRACE_LENGTH

        # input을 모두 tl=10이 되도록 padding 처리 => LSTM 입력을 위해서
        if tl < MAX_TRACE_LENGTH:
            state_input_org = [state_input[index][i] for i in range(0, len(state_input[index]))]
            reward_org = [reward[index][0][i] for i in range(0, len(reward[index][0]))]
            state_input_update = padding_hybrid_feature_input(state_input_org)
            state_input[index] = state_input_update
            reward_update = padding_hybrid_reward(reward_org)
            reward[index] = reward_update
        
        state_trace_length_output.append(tl)
    return state_trace_length_output, state_input, reward
