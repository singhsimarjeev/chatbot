# -*- coding: utf-8 -*-
import os

os.environ['KERAS_BACKEND'] = 'theano'

from keras.layers import Input, Embedding, LSTM, Dense, RepeatVector, Dropout, merge
from keras.optimizers import Adam 
from keras.models import Model
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.preprocessing import sequence
from keras.layers import concatenate

import keras.backend as K
import numpy as np
np.random.seed(1234)  # for reproducibility
import cPickle
import theano
import os.path
import sys
import nltk
import re
import time

nltk.download('punkt')


word_embedding_size = 100

sentence_embedding_size = 300

dictionary_size = 7000

maxlen_input = 50

vocabulary_file = 'vocabulary_extra'
weights_file = 'model_weights_final.h5'
unknown_token = 'random'


def greedy_decoder(input):

    flag = 0

    prob = 1

    partial = np.zeros((1,maxlen_input))

    partial[0, -1] = 2  

    for k in range(maxlen_input - 1):

        tmp1 = model.predict([input, partial])
        tmp = ye[0,:]
        p = np.max(tmp)
        mp = np.argmax(tmp1)
        partial[0, 0:-1] = partial[0, 1:]
        partial[0, -1] = mp
        if mp == 3:  
            flag = 1
        if flag == 0:    
            prob = prob * p

    text = ''

    for k in ans_partial[0]:
        k = k.astype(int)
        if k < (dictionary_size-2):
            w = vocabulary[k]
            text = text + w[0] + ' '
    return(text, prob)
    
    
def preprocess(raw, val):
    
    l1 = ['won’t','won\'t','wouldn’t','wouldn\'t','’m', '’re', '’ve', '’ll', '’s','’d', 'n’t', '\'m', '\'re', '\'ve', '\'ll', '\'s', '\'d', 'can\'t', 'n\'t', 'B: ', 'A: ', ',', ';', '.', '?', '!', ':', '. ?', ',   .', '. ,', 'EOS', 'BOS', 'eos', 'bos']
    l2 = ['will not','will not','would not','would not',' am', ' are', ' have', ' will', ' is', ' had', ' not', ' am', ' are', ' have', ' will', ' is', ' had', 'can not', ' not', '', '', ' ,', ' ;', ' .', ' ?', ' !', ' :', '? ', '.', ',', '', '', '', '']
    l3 = ['-', '_', ' *', ' /', '* ', '/ ', '\"', ' \\"', '\\ ', '--', '...', '. . .']

    raw_word = raw.lower()

    for j, term in enumerate(l1):
        raw_word = raw.replace(term,l2[j])
        
    for term in l3:
        raw_word = raw.replace(term,' ')

    return raw_word

def tokenize(sentences):

    # Tokenizing the sentences into words:
    tokenized_sentences = nltk.word_tokenize(sentences.decode('utf-8'))

    index_to_word = [x[0] for x in vocabulary]
    word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])
    tokenized_sentences = [w if w in word_to_index else unknown_token for w in tokenized_sentences]

    tmp = np.asarray([word_to_index[w] for w in tokenized_sentences])
    s = tmp.size
    final_token = np.zeros((1,maxlen_input))
    if s < (maxlen_input + 1):
        final_token[0,- s:] = tmp
    else:
        final_token[0,:] = tmp[- maxlen_input:]
    
    return final_token




ad = Adam(lr=0.00005) 

input_context = Input(shape=(maxlen_input,), dtype='int32', name='context')
input_answer = Input(shape=(maxlen_input,), dtype='int32')
LSTM_encoder = LSTM(sentence_embedding_size, kernel_initializer= 'lecun_uniform', name='Encode context')
LSTM_decoder = LSTM(sentence_embedding_size, kernel_initializer= 'lecun_uniform')

if os.path.isfile(weights_file):
    Shared_Embedding = Embedding(output_dim=word_embedding_size, input_dim=dictionary_size, input_length=maxlen_input, name='Shared')
else:
    Shared_Embedding = Embedding(output_dim=word_embedding_size, input_dim=dictionary_size, weights=[embedding_matrix], input_length=maxlen_input, name='Shared')
word_embedding_context = Shared_Embedding(input_context)
context_embedding = LSTM_encoder(word_embedding_context)

word_embedding_answer = Shared_Embedding(input_answer)
answer_embedding = LSTM_decoder(word_embedding_answer)

merge_layer = concatenate([context_embedding, answer_embedding], axis=1, name='concatenate the embeddings of the context and the answer up to current token')
out = Dense(dictionary_size/2, activation="relu", name='relu activation')(merge_layer)
out = Dense(dictionary_size, activation="softmax", name='likelihood of the current token using softmax activation')(out)

model = Model(inputs=[input_context, input_answer], outputs = [out])

model.compile(loss='categorical_crossentropy', optimizer=ad)


if os.path.isfile(weights_file):
    model.load_weights(weights_file)


# Loading the data:
vocabulary = cPickle.load(open(vocabulary_file, 'rb'))


# Processing human:

prob = 0
que = ''
last_query  = ' '
last_last_query = ''
text = ' '
last_text = ''


while que <> 'exit .':
    
    que = raw_input('Human: ')
    que = preprocess(que, 'Simarjeev')
    # Collecting data for training:
    q = last_query + ' ' + text
    a = que
    qf.write(q + '\n')
    af.write(a + '\n')
    # Composing the context:
    if prob > 0.2:
        query = text + ' ' + que
    else:    
        query = que
   
    last_text = text
    
    Q = tokenize(query)
    
    
    predout, prob = greedy_decoder(Q[0:1])
    start_index = predout.find('EOS')
    text = preprocess(predout[0:start_index], name)
    print ('Machine: ' + text)
    
    last_last_query = last_query    
    last_query = que


qf.close()
af.close()

