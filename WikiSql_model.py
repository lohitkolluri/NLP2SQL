#pip install tensorflow pandas scikit-learn matplotlib seaborn
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Attention, Concatenate, Dropout, Bidirectional
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
def load_wikisql_data(file_path):
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data):
    questions = data['question'].astype(str).tolist()
    sql_queries = data['sql'].astype(str).tolist()
    return questions, sql_queries

def tokenize_and_pad(texts, tokenizer, max_length):
    sequences = tokenizer.texts_to_sequences(texts)
    return pad_sequences(sequences, maxlen=max_length, padding='post')
def create_model(vocab_size, embedding_dim, rnn_units, dropout_rate=0.5, l2_reg=0.001):
    question_input = Input(shape=(None,), name='question_input')
    sql_input = Input(shape=(None,), name='sql_input')

    embedding_layer = Embedding(vocab_size, embedding_dim)
    question_emb = embedding_layer(question_input)
    sql_emb = embedding_layer(sql_input)

    lstm_question = Bidirectional(LSTM(rnn_units, return_sequences=True, kernel_regularizer=l2(l2_reg)))(question_emb)
    lstm_question = Dropout(dropout_rate)(lstm_question)
    lstm_sql = Bidirectional(LSTM(rnn_units, return_sequences=True, kernel_regularizer=l2(l2_reg)))(sql_emb)
    lstm_sql = Dropout(dropout_rate)(lstm_sql)

    attention = Attention()([lstm_question, lstm_sql])
    concat = Concatenate()([lstm_question, attention])
    output = Dense(vocab_size, activation='softmax', kernel_regularizer=l2(l2_reg))(concat)

    model = Model(inputs=[question_input, sql_input], outputs=output)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model
def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
lr_scheduler = LearningRateScheduler(scheduler)
# Load data
train_data = load_wikisql_data('train.csv')
val_data = load_wikisql_data('validation.csv')
test_data = load_wikisql_data('test.csv')

train_questions, train_sql_queries = preprocess_data(train_data)
val_questions, val_sql_queries = preprocess_data(val_data)
test_questions, test_sql_queries = preprocess_data(test_data)

# Tokenization and padding
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_questions + train_sql_queries)
vocab_size = len(tokenizer.word_index) + 1

max_length = 50
train_questions_seq = tokenize_and_pad(train_questions, tokenizer, max_length)
train_sql_queries_seq = tokenize_and_pad(train_sql_queries, tokenizer, max_length)
val_questions_seq = tokenize_and_pad(val_questions, tokenizer, max_length)
val_sql_queries_seq = tokenize_and_pad(val_sql_queries, tokenizer, max_length)

# Define hyperparameters
embedding_dim = 300
rnn_units = 512
dropout_rate = 0.3
l2_reg = 0.0005

# Create and train the model
model = create_model(vocab_size, embedding_dim, rnn_units, dropout_rate, l2_reg)

history = model.fit(
    [train_questions_seq, train_sql_queries_seq],
    np.expand_dims(train_sql_queries_seq, -1),
    epochs=5,
    validation_data=([val_questions_seq, val_sql_queries_seq], np.expand_dims(val_sql_queries_seq, -1)),
    callbacks=[early_stopping, lr_scheduler]
)

# Plot training history
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()

# Evaluate the model
test_questions_seq = tokenize_and_pad(test_questions, tokenizer, max_length)
test_sql_queries_seq = tokenize_and_pad(test_sql_queries, tokenizer, max_length)

loss = model.evaluate([test_questions_seq, test_sql_queries_seq], np.expand_dims(test_sql_queries_seq, -1))
predictions = model.predict([test_questions_seq, test_sql_queries_seq])

# Convert predictions
predicted_sql_queries = np.argmax(predictions, axis=-1)

# Calculate exact match accuracy
exact_matches = 0
for true_query, pred_query in zip(test_sql_queries_seq, predicted_sql_queries):
    if np.array_equal(true_query, pred_query):
        exact_matches += 1

exact_match_accuracy = exact_matches / len(test_sql_queries_seq)

print(f"Loss: {loss}")
print(f"Exact Match Accuracy: {exact_match_accuracy * 100:.2f}%")
def predict_sql(natural_language_input):
    input_seq = tokenize_and_pad([natural_language_input], tokenizer, max_length)
    prediction = model.predict([input_seq, input_seq])
    predicted_seq = tf.argmax(prediction, axis=-1).numpy()[0]
    sql_query = tokenizer.sequences_to_texts([predicted_seq])
    return sql_query[0]

# Example prediction

example_question = test_questions[1]
predicted_sql = predict_sql(example_question)
print(f"Predicted SQL Query: {predicted_sql}")
