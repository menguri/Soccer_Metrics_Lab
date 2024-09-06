import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# 문제를 진단하기 위해 anomaly detection 활성화
torch.autograd.set_detect_anomaly(True)

class TD_Prediction_TT_Embed(nn.Module):
    def __init__(self, feature_number, h_size, max_trace_length, learning_rate,
                 output_layer_size=3, lstm_layer_num=2, dense_layer_num=2, model_name="tt_lstm"):
        super(TD_Prediction_TT_Embed, self).__init__()
        
        self.feature_number = feature_number
        self.h_size = h_size
        self.max_trace_length = max_trace_length
        self.learning_rate = learning_rate
        self.lstm_layer_num = lstm_layer_num
        self.dense_layer_num = dense_layer_num
        self.output_layer_size = output_layer_size
        self.model_name = model_name
        
        # Home LSTM layers
        self.lstm_home = nn.LSTM(input_size=feature_number, hidden_size=h_size, num_layers=lstm_layer_num, batch_first=True)
        
        # Away LSTM layers
        self.lstm_away = nn.LSTM(input_size=feature_number, hidden_size=h_size, num_layers=lstm_layer_num, batch_first=True)
        
        # Embedding layers
        self.embed_home = nn.Linear(h_size, h_size)
        self.embed_away = nn.Linear(h_size, h_size)
        
        # Dense layers
        self.dense_layers = nn.ModuleList()
        for i in range(dense_layer_num):
            input_size = h_size if i == 0 else h_size
            output_size = h_size if i < dense_layer_num - 1 else output_layer_size
            self.dense_layers.append(nn.Linear(input_size, output_size))
        
        # Output activation
        self.softmax = nn.Softmax(dim=1)
        
        # Loss function
        self.loss_fn = nn.MSELoss()
        
        # Separate optimizers for Home and Away including Dense layers
        self.optimizer_home = optim.Adam(list(self.lstm_home.parameters()) + 
                                         list(self.embed_home.parameters()) + 
                                         list(self.dense_layers.parameters()), lr=learning_rate)
                                         
        self.optimizer_away = optim.Adam(list(self.lstm_away.parameters()) + 
                                         list(self.embed_away.parameters()) + 
                                         list(self.dense_layers.parameters()), lr=learning_rate)
    
    def forward(self, rnn_input, trace_lengths, home_away_indicator):
        # Home LSTM
        packed_input = nn.utils.rnn.pack_padded_sequence(rnn_input, trace_lengths, batch_first=True, enforce_sorted=False)
        packed_output_home, _ = self.lstm_home(packed_input)
        output_home, _ = nn.utils.rnn.pad_packed_sequence(packed_output_home, batch_first=True)
        
        # Away LSTM
        packed_output_away, _ = self.lstm_away(packed_input)
        output_away, _ = nn.utils.rnn.pad_packed_sequence(packed_output_away, batch_first=True)
        
        # Get the last relevant output
        last_output_home = output_home[range(output_home.shape[0]), trace_lengths - 1]
        last_output_away = output_away[range(output_away.shape[0]), trace_lengths - 1]
        
        # Embedding layers
        home_embed = self.embed_home(last_output_home)
        away_embed = self.embed_away(last_output_away)
        
        # Select between home and away embeddings
        embed_layer = torch.where(home_away_indicator.unsqueeze(-1), home_embed, away_embed)
        
        # Dense layers
        x = embed_layer
        for dense_layer in self.dense_layers[:-1]:
            x = torch.relu(dense_layer(x))
        
        # Final dense layer with softmax activation
        x = self.dense_layers[-1](x)
        x = self.softmax(x)
        
        return x
    
    def compute_loss(self, predictions, targets, home_away_indicator, n):
        if n == 0:
            loss = self.loss_fn(predictions[home_away_indicator], targets[home_away_indicator])
        else:
            loss = self.loss_fn(predictions[~home_away_indicator], targets[~home_away_indicator])
        return loss
    
    def train_step(self, rnn_input, trace_lengths, home_away_indicator, y):
        # Transition의 Loss는 한 개의 tower에만 전달된다.
        # Compute gradients and update for Home tower
        # predictions = self.forward(rnn_input, trace_lengths, home_away_indicator)

        # Zero out gradients for both optimizers
        self.optimizer_home.zero_grad()
        self.optimizer_away.zero_grad()
        
        # Compute gradients and update for Away tower
        predictions = self.forward(rnn_input, trace_lengths, home_away_indicator)
        away_loss = self.compute_loss(predictions, y, home_away_indicator, 1)
        away_loss.backward(retain_graph=True)
        self.optimizer_away.step()

        # Compute gradients and update for Home tower
        predictions = self.forward(rnn_input, trace_lengths, home_away_indicator)
        home_loss = self.compute_loss(predictions, y, home_away_indicator, 0)        
        home_loss.backward()   
        self.optimizer_home.step()
        
        return home_loss.item(), away_loss.item(), predictions

