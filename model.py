import tensorflow
from tensorflow.keras import Sequential, Model, Input
from tensorflow.keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
#https://www.tensorflow.org/install/gpu#software_requirements
#https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html
from numpy.random import seed
seed(1)
tensorflow.random.set_seed(2)

import keras as k

def get_bilstm_lstm_model(input_dim, output_dim, input_length, n_tags):
    model = Sequential()

    # Add Embedding layer
    model.add(Embedding(input_dim=input_dim, output_dim=output_dim, input_length=input_length))

    # Add bidirectional LSTM
    #model.add(Bidirectional(LSTM(units=output_dim, return_sequences=True, dropout=0.2, recurrent_dropout=0.2), merge_mode = 'concat'))
    model.add(Bidirectional(LSTM(units=output_dim, return_sequences=True, dropout=0.2, recurrent_dropout=0.2), merge_mode = 'concat'))

    # Add LSTM
    model.add(LSTM(units=output_dim*4, return_sequences=True, dropout=0.5, recurrent_dropout=0.5, activation='tanh'))
    #model.add(LSTM(units=output_dim*4, return_sequences=True, dropout=0.5, recurrent_dropout=0.5))

    # Add timeDistributed Layer
    model.add(TimeDistributed(Dense(n_tags, activation="relu")))

    #Optimiser 
    adam = k.optimizers.Adam(learning_rate=0.0005, beta_1=0.9, beta_2=0.999)

    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer = adam, metrics=['accuracy'])
    model.summary()
    
    return model

def train_model(X, y, model):
    loss = list()
    for _ in range(5):#25
        # fit model for one epoch on this sequence
        hist = model.fit(X, y, batch_size=128, verbose=1, epochs=1, validation_split=0.2)
        print(hist.history['loss'][0])
        loss.append(hist.history['loss'][0])
    return loss
