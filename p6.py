import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.datasets import imdb

# Load and preprocess IMDb dataset
vocab_size, max_length = 10000, 100
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=vocab_size)
train_padded = pad_sequences(train_data, maxlen=max_length, padding='post', truncating='post')
test_padded = pad_sequences(test_data, maxlen=max_length, padding='post', truncating='post')

# Build and compile the model
model = Sequential([
    Embedding(vocab_size, 32, input_length=max_length),
    LSTM(64), Dropout(0.5),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train and evaluate the model
model.fit(train_padded, train_labels, epochs=10, validation_data=(test_padded, test_labels))
loss, accuracy = model.evaluate(test_padded, test_labels)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
