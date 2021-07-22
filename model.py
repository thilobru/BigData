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
    # model.add(LSTM(units=output_dim, return_sequences=True, dropout=0.5, recurrent_dropout=0.5))
    model.add(LSTM(units=output_dim, return_sequences=True, dropout=0.5, recurrent_dropout=0.5))

    # Add timeDistributed Layer
    # model.add(TimeDistributed(Dense(n_tags, activation="relu")))
    model.add(TimeDistributed(Dense(n_tags, activation="softmax")))

    #Optimiser 
    adam = k.optimizers.Adam(learning_rate=0.0005, beta_1=0.9, beta_2=0.999)

    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer = adam, metrics=['accuracy'], sample_weight_mode='temporal')
    model.summary()
    
    return model
## class weights berechnen


def generate_class_weights(class_series, multi_class=True, one_hot_encoded=False):
    import numpy as np
    from sklearn.utils.class_weight import compute_class_weight
    from sklearn.preprocessing import MultiLabelBinarizer
    """
    Method to generate class weights given a set of multi-class or multi-label labels, both one-hot-encoded or not.
    Some examples of different formats of class_series and their outputs are:
        - generate_class_weights(['mango', 'lemon', 'banana', 'mango'], multi_class=True, one_hot_encoded=False)
        {'banana': 1.3333333333333333, 'lemon': 1.3333333333333333, 'mango': 0.6666666666666666}
        - generate_class_weights([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]], multi_class=True, one_hot_encoded=True)
        {0: 0.6666666666666666, 1: 1.3333333333333333, 2: 1.3333333333333333}
        - generate_class_weights([['mango', 'lemon'], ['mango'], ['lemon', 'banana'], ['lemon']], multi_class=False, one_hot_encoded=False)
        {'banana': 1.3333333333333333, 'lemon': 0.4444444444444444, 'mango': 0.6666666666666666}
        - generate_class_weights([[0, 1, 1], [0, 0, 1], [1, 1, 0], [0, 1, 0]], multi_class=False, one_hot_encoded=True)
        {0: 1.3333333333333333, 1: 0.4444444444444444, 2: 0.6666666666666666}
    The output is a dictionary in the format { class_label: class_weight }. In case the input is one hot encoded, the class_label would be index
    of appareance of the label when the dataset was processed. 
    In multi_class this is np.unique(class_series) and in multi-label np.unique(np.concatenate(class_series)).
    Author: Angel Igareta (angel@igareta.com)
    """
    if multi_class:
        # If class is one hot encoded, transform to categorical labels to use compute_class_weight   
        if one_hot_encoded:
            class_series = np.argmax(class_series, axis=1)
  
        # Compute class weights with sklearn method
        class_labels = np.unique(class_series)
        class_weights = compute_class_weight(class_weight='balanced', classes=class_labels, y=class_series)
        return dict(zip(class_labels, class_weights))
    else:
        # It is neccessary that the multi-label values are one-hot encoded
        mlb = None
        if not one_hot_encoded:
            mlb = MultiLabelBinarizer()
            class_series = mlb.fit_transform(class_series)

        n_samples = len(class_series)
        n_classes = len(class_series[0])

        # Count each class frequency
        class_count = [0] * n_classes
        for classes in class_series:
            for index in range(n_classes):
                if classes[index] != 0:
                    class_count[index] += 1
        
        # Compute class weights using balanced method
        class_weights = [n_samples / (n_classes * freq) if freq > 0 else 1 for freq in class_count]
        class_labels = range(len(class_weights)) if mlb is None else mlb.classes_
        return dict(zip(class_labels, class_weights))

# ToDo: Array korreckt befüllen:
# https://github.com/keras-team/keras/issues/3653#issuecomment-761085597
def generate_sample_weights(training_data, class_weights): 
    import numpy as np
    #replaces values for up to 3 classes with the values from class_weights#
    # ToDo: Train Tags sind OnHot Encodings 
    sample_weights = [np.where(y==0,class_weights[0],
                        np.where(y==1,class_weights[1],
                        np.where(y==2,class_weights[2],
                        np.where(y==3,class_weights[3],
                        np.where(y==4,class_weights[4],
                        np.where(y==3,class_weights[5],
                        np.where(y==4,class_weights[6],
                        np.where(y==5,class_weights[7])))))))) for y in training_data]
    return np.asarray(sample_weights)

def train_model(X, y, model):
    #class_weights = generate_class_weights(y[0], multi_class=True, one_hot_encoded=True)
    #class_weights = generate_sample_weights(y, class_weights)
    import numpy as np
    class_weights = np.zeros((39010, 300))
    # ToDo: Array korreckt befüllen:
    # https://github.com/keras-team/keras/issues/3653#issuecomment-761085597
    class_weights[:, 0] += 0.1
    class_weights[:, 1] += 42.0
    class_weights[:, 2] += 42.0
    class_weights[:, 3] += 42.0
    class_weights[:, 4] += 42.0
    class_weights[:, 5] += 42.0
    class_weights[:, 6] += 42.0
    class_weights[:, 7] += 42.0
    loss = list()
    for _ in range(5):#25
        # fit model for one epoch on this sequence
        hist = model.fit(X, y, batch_size=512, verbose=1, epochs=1, validation_split=0.2, sample_weight=class_weights)
        print(hist.history['loss'][0])
        loss.append(hist.history['loss'][0])
    return model, loss
