#classification of newswires a multiclass classification reuteras dataset
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, utils
from tensorflow.keras.datasets import reuters

# Load and preprocess data
(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)

# Vectorize sequences
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

x_train, x_test = vectorize_sequences(train_data), vectorize_sequences(test_data)

# One-hot encode labels
one_hot_train_labels = utils.to_categorical(train_labels)
one_hot_test_labels = utils.to_categorical(test_labels)

# Build model
model = tf.keras.Sequential([
    layers.Dense(64, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(46, activation='softmax')
])

# Compile model
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# Train-validation split
x_val, partial_x_train = x_train[:1000], x_train[1000:]
y_val, partial_y_train = one_hot_train_labels[:1000], one_hot_train_labels[1000:]

# Train model
history = model.fit(partial_x_train, partial_y_train, epochs=25, batch_size=512, validation_data=(x_val, y_val))

# Evaluate model
results = model.evaluate(x_test, one_hot_test_labels)
