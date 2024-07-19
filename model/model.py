from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers


def create_model(vocab_size, embedding_dim, input_length):
  model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=input_length),
        Bidirectional(LSTM(256, return_sequences = True)),
        Dropout(0.2),
        LSTM(128),
        Dense(vocab_size//2, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        Dense(vocab_size , activation='softmax')
      ])
  return model
