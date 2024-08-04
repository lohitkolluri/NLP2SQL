#!/usr/bin/env python
# coding: utf-8

# In[2]:


#get_ipython().system(' pip install kaggle')

# In[3]:


#get_ipython().system(' kaggle datasets download -d shahrukhkhan/wikisql')

# In[4]:


#get_ipython().system(" unzip 'wikisql.zip'")

# In[5]:


#get_ipython().system(' pip install tensorflow')

# In[6]:


#get_ipython().system(' pip install seaborn')

# In[7]:


import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Attention, Concatenate
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# In[8]:


def load_wikisql_data(file_path):
    data = pd.read_csv(file_path)
    return data


# In[9]:


def preprocess_data(data):
    questions = data['question'].astype(str).tolist()
    sql_queries = data['sql'].astype(str).tolist()
    return questions, sql_queries


# In[10]:


train_data = load_wikisql_data('train.csv')
val_data = load_wikisql_data('validation.csv')
test_data = load_wikisql_data('test.csv')

# In[11]:


print(train_data)

# In[12]:


train_questions, train_sql_queries = preprocess_data(train_data)
val_questions, val_sql_queries = preprocess_data(val_data)
test_questions, test_sql_queries = preprocess_data(test_data)

# In[13]:


tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_questions + train_sql_queries)
vocab_size = len(tokenizer.word_index) + 1


def tokenize_and_pad(texts, max_length):
    sequences = tokenizer.texts_to_sequences(texts)
    return pad_sequences(sequences, maxlen=max_length, padding='post')


max_length = 50
train_questions_seq = tokenize_and_pad(train_questions, max_length)
train_sql_queries_seq = tokenize_and_pad(train_sql_queries, max_length)
val_questions_seq = tokenize_and_pad(val_questions, max_length)
val_sql_queries_seq = tokenize_and_pad(val_sql_queries, max_length)

# In[14]:


from tensorflow.keras.layers import Dropout, LSTM, Dense, Input, Embedding, Attention, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2  # Correct import for L2 regularization


def create_model(vocab_size, embedding_dim, rnn_units, dropout_rate=0.5, l2_reg=0.001):
    # Define inputs
    question_input = Input(shape=(None,), name='question_input')
    sql_input = Input(shape=(None,), name='sql_input')

    # Embedding layers
    embedding_layer = Embedding(vocab_size, embedding_dim)
    question_emb = embedding_layer(question_input)
    sql_emb = embedding_layer(sql_input)

    # LSTM layers with Dropout and L2 regularization
    lstm_question = LSTM(rnn_units, return_sequences=True, kernel_regularizer=l2(l2_reg))(question_emb)
    lstm_question = Dropout(dropout_rate)(lstm_question)
    lstm_sql = LSTM(rnn_units, return_sequences=True, kernel_regularizer=l2(l2_reg))(sql_emb)
    lstm_sql = Dropout(dropout_rate)(lstm_sql)

    # Attention layer
    attention = Attention()([lstm_question, lstm_sql])

    # Concatenate attention output with LSTM output
    concat = Concatenate()([lstm_question, attention])

    # Dense layer for output
    output = Dense(vocab_size, activation='softmax', kernel_regularizer=l2(l2_reg))(concat)

    model = Model(inputs=[question_input, sql_input], outputs=output)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


# Define hyperparameters
embedding_dim = 300  # Updated embedding dimension
rnn_units = 512  # Updated number of RNN units
dropout_rate = 0.3  # Updated dropout rate
l2_reg = 0.0005  # Updated L2 regularization

# Create the model
model = create_model(vocab_size, embedding_dim, rnn_units, dropout_rate, l2_reg)

# In[16]:


history = model.fit(
    [train_questions_seq, train_sql_queries_seq],
    np.expand_dims(train_sql_queries_seq, -1),  # Adjust dimensions for sparse_categorical_crossentropy
    epochs=1,
    validation_data=([val_questions_seq, val_sql_queries_seq], np.expand_dims(val_sql_queries_seq, -1))
)

# In[17]:


from sklearn.metrics import accuracy_score
import numpy as np

# Tokenize and pad the test data
test_questions_seq = tokenize_and_pad(test_questions, max_length)
test_sql_queries_seq = tokenize_and_pad(test_sql_queries, max_length)

# Evaluate the model on the test data
loss = model.evaluate([test_questions_seq, test_sql_queries_seq], np.expand_dims(test_sql_queries_seq, -1))

# Generate predictions
predictions = model.predict([test_questions_seq, test_sql_queries_seq])

# Convert predictions to the same format as ground truth for comparison
# This conversion depends on your specific output format (e.g., token ids, one-hot encoding)
# Here, assuming predictions are token ids, we round and convert to integers
predicted_sql_queries = np.argmax(predictions, axis=-1)

# Calculate exact match accuracy
exact_matches = 0
for true_query, pred_query in zip(test_sql_queries_seq, predicted_sql_queries):
    if np.array_equal(true_query, pred_query):
        exact_matches += 1

exact_match_accuracy = exact_matches / len(test_sql_queries_seq)

print(f"Loss: {loss}")
print(f"Exact Match Accuracy: {exact_match_accuracy * 100:.2f}%")


# In[ ]:


def predict_sql(natural_language_input):
    input_seq = tokenize_and_pad([natural_language_input], max_length)
    prediction = model.predict([input_seq, input_seq])  # Need both inputs for the model
    predicted_seq = tf.argmax(prediction, axis=-1).numpy()[0]
    sql_query = tokenizer.sequences_to_texts([predicted_seq])
    return sql_query[0]


# In[ ]:


print(test_questions[1])

# In[ ]:


example_question = test_questions[1]
predicted_sql = predict_sql(example_question)
print(f"Predicted SQL Query: {predicted_sql}")

# In[ ]:
