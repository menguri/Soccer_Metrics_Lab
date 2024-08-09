import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


class td_prediction_lstm_V4:
    def __init__(self, FEATURE_NUMBER, H_SIZE, MAX_TRACE_LENGTH, learning_rate, rnn_type='bp_last_step'):
        """
        define a dynamic LSTM
        """
        # LSTM Layer
        rnn_input = Input(shape=(10, FEATURE_NUMBER), name="x_1")
        trace_lengths = Input(shape=(1,), dtype=tf.int32, name="tl")

        lstm_layer = LSTM(H_SIZE * 2, return_sequences=True, return_state=True, kernel_initializer=tf.random_uniform_initializer(-0.05, 0.05))
        rnn_output, state_h, state_c = lstm_layer(rnn_input)

        # outputs = tf.stack(rnn_output)
        outputs = rnn_output

        # Hack to build the indexing and retrieve the right output.
        batch_size = tf.shape(outputs)[0]
        # Start indices for each sample
        index = tf.range(0, batch_size) * MAX_TRACE_LENGTH + (tf.squeeze(trace_lengths) - 1)
        # Indexing
        rnn_last = tf.gather(tf.reshape(outputs, [-1, H_SIZE * 2]), index)

        num_layer_1 = H_SIZE * 2
        num_layer_2 = 1000
        num_layer_3 = 3

        # Dense Layer first
        activation1 = Dense(num_layer_2, activation='relu', kernel_initializer=tf.keras.initializers.GlorotUniform(), name="Dense_Layer_first")(rnn_last)

        # Dense Layer second
        read_out = Dense(num_layer_3, kernel_initializer=tf.keras.initializers.GlorotUniform(), name="Dense_Layer_second")(activation1)

        self.model = Model(inputs=[rnn_input, trace_lengths], outputs=read_out)

        # Placeholder for the true output
        self.y = tf.keras.Input(shape=(num_layer_3,), name="y")

        # Cost calculation
        self.cost = tf.reduce_mean(tf.square(self.y - read_out))
        self.diff = tf.reduce_mean(tf.abs(self.y - read_out))

        # Optimizer
        self.train_step = Adam(learning_rate=learning_rate).minimize(self.cost)

        # Compile model
        self.model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error')

    def train(self, x, tl, y):
        self.model.fit([x, tl], y, epochs=1)
