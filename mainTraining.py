import pandas as pd
import dict_map as dm

df = pd.read_csv('Daten/trainingdata300.csv',escapechar="\\",sep=",",error_bad_lines=False,warn_bad_lines=False)
print(df)
# df = df.head(1000)
token2idx, idx2token = dm.get_dict_map(df, 'token')
tag2idx, idx2tag = dm.get_dict_map(df, 'tag')

df['Word_idx'] = df['Wort'].map(token2idx)
df['Tag_idx'] = df['Attribut'].map(tag2idx)
print(df)

df_group = df.groupby(by = ['satzId'], as_index=False)['Wort', 'Attribut', 'Word_idx', 'Tag_idx'].agg(lambda x: list(x))
#df_group
import pad_train_test as ptt
train_tokens, val_tokens, test_tokens, train_tags, val_tags, test_tags = ptt.get_pad_train_test_val(df_group, df, tag2idx)

input_dim = len(list(set(df['Wort'].to_list())))+1
output_dim = 32
input_length = max([len(s) for s in df_group['Word_idx'].tolist()])
n_tags = len(tag2idx)
print('input_dim: ', input_dim, '\noutput_dim: ', output_dim, '\ninput_length: ', input_length, '\nn_tags: ', n_tags)

import model as md
model_bilstm_lstm = md.get_bilstm_lstm_model(input_dim, output_dim, input_length, n_tags)

#choco install graphviz, pip install pydot
#from tensorflow.keras.utils import plot_model 
#plot_model(model_bilstm_lstm)

import numpy as np
results = pd.DataFrame()
model, results['with_add_lstm'] = md.train_model(train_tokens, np.array(train_tags), model_bilstm_lstm)
print(results)

# Save the model
filepath = './models/train300wWeights'
# save_model(md, filepath)
model.save(filepath)