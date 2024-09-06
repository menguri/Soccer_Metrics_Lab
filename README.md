# Goal Impact Metric : Sarsa + LSTM

This repository is an implementation of the model in the paper. "[Deep soccer analytics: learning an action-value function for evaluating soccer players](https://link.springer.com/article/10.1007/s10618-020-00705-9)". I searched the paper author's github, but couldn't find the code for the model, so I implemented it myself.


## GIM's Model <br>
1-1. python td_three_prediction_lstm.py >> start to train! <br>
1-2. utils.py is tools library for manufacturing state_input <br>
1-3. nn/td_two_tower_lstm.py : implementing two tower lstm <br>
1-4. preprocess_data.py, labels.py are for fitting raw data format of paper <br> 

<br><br>

## How to use Soccer_Metrics_Lab <br>
[1] Check data-fifia(you can get this folder from vaep/notebook/1&2 notebook) | gim | socceraction | vaep <br>
[2] start gim/notebook/1.notebook and make datastore folder <br>
[3] train GIM model through starting td_three_prediction_lstm.py <br>
[4] explicit GIM Metric from gim/notebook/3.notebook <br>
