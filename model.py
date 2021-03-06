import tensorflow as tf
from tensorflow.keras import Sequential, Model, Input
from tensorflow.keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
#https://www.tensorflow.org/install/gpu#software_requirements
#https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html
from numpy.random import seed
seed(1)
tf.random.set_seed(2)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import MultiLabelBinarizer

import keras as k
import numpy as np

def generate_class_weights(class_series, multi_class=True, one_hot_encoded=False):
    array = []
    #for x in range(0, 20000):
    for x in  range(0, len(class_series)):
        for y in range(0, 300):
            array.append(class_series[x][y])
    class_series = array

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
        print(dict(zip(class_labels, class_weights)))
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
        print(dict(zip(class_labels, class_weights)))
        return dict(zip(class_labels, class_weights))

# ToDo: Array korreckt bef??llen:
# https://github.com/keras-team/keras/issues/3653#issuecomment-761085597
def generate_sample_weights(train_tags, class_weights): 
    #replaces values for up to 7 classes with the values from class_weights#
    # ToDo: Train Tags sind OnHot Encodings 
    train_tags = np.argmax(train_tags, axis=1)
    sample_weights = [np.where(y==0,class_weights[0],
                        np.where(y==1,class_weights[1],
                        np.where(y==2,class_weights[2],
                        np.where(y==3,class_weights[3],
                        np.where(y==4,class_weights[4],
                        np.where(y==5,class_weights[5],
                        np.where(y==6,class_weights[6],
                        y))))))) for y in train_tags]
    return np.asarray(sample_weights)

def get_bilstm_lstm_model(input_dim, output_dim, input_length, n_tags):
    model = Sequential()
    # Add Embedding layer
    model.add(Embedding(input_dim=input_dim, output_dim=output_dim, input_length=input_length))
    # Add bidirectional LSTM
    model.add(Bidirectional(LSTM(units=output_dim, return_sequences=True, dropout=0.2, recurrent_dropout=0.2), merge_mode = 'concat'))
    # Add LSTM
    model.add(LSTM(units=output_dim, return_sequences=True, dropout=0.5, recurrent_dropout=0.5))
    # Add timeDistributed Layer
    # model.add(TimeDistributed(Dense(n_tags, activation="relu"))) # doesnt calculate loss
    model.add(TimeDistributed(Dense(n_tags, activation="softmax")))
    #Optimiser 
    adam = k.optimizers.Adam(learning_rate=0.0005, beta_1=0.9, beta_2=0.999)
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer = adam, metrics=['accuracy'], sample_weight_mode='temporal')
    model.summary()
    
    return model

def train_model(X, y, Xv, yv, model):
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0.0005, #default: 0.0001 -> wenn h??her, stoppt es eher (nach 11/12 iter), bei 0.00001: 31 iter
        patience=5,
        verbose=1, 
        mode='min',
        restore_best_weights=True)

    class_weights = generate_class_weights(y, multi_class=True, one_hot_encoded=True)
    sample_weights = np.zeros((len(y), len(X[0])))
    for x in range(0, len(y)-1):
        sample_weights[x] = generate_sample_weights(y[x], class_weights)
            
    hist = model.fit(X, y, 
        validation_data=(Xv, yv),
        batch_size=64, 
        epochs=50,
        verbose=1, 
        callbacks=[early_stop], 
        sample_weight=sample_weights)
    print(hist.history['loss'])
    return model, hist
