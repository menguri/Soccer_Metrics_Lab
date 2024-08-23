# Soccer_Metrics_Lab

#### STEP 1. xT, VAEP, GIM 등 다양한 metrics 구현

##### GIM's Model 설명 <br>
1-1. python td_three_prediction_lstm.py >> 훈련 시작 <br>
1-2. tuils.py는 state_input 구조 제작, nn/td_two_tower_lstm.py는 two tower lstm을 구현한 파일 <br>
1-3. preprocess_data.py, labels.py는 데이터 전처리 utils <br> 

##### Soccer_Metrics_Lab 사용법 <br>
[1] data-fifia(h5 파일 저장 폴더) | gim | socceraction | vaep 폴더 존재 확인 <br>
[2] gim/notebook에서 1번 노트북 실행하여 datastore 파일 생성 <br>
[3] td_three_prediction_lstm.py 를 실행하여 모델 학습 <br>
[4] gim/notebook에서 3번 노트북 실행하여 GIM metric 뽑아내기 <br>


#### STEP 2. metrics들 간의 상관관계 분석
#### STEP 3. 논문 'Towards maximizing expected possession outcome in soccer'의 모델을 참고하여 성능 비교 분석
#### STEP 4. 관련 데이터 및 영상 링크: https://drive.google.com/drive/folders/1jTZC2p1pMSmb4D1A_F8j-4wzS-YYrbYr?usp=drive_link