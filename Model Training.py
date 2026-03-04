# Library imports
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import os

# Disable GPU computing
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Enable GPU computing
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# Set dark background style
plt.style.use('dark_background')

# Function to load data from numpy file
def load_data(filepath='data.npz'):
    """Load preprocessed data and labels from a numpy file."""
    if os.path.exists(filepath):
        loaded = np.load(filepath)
        X_data = loaded['X_data']
        Y_data = loaded['Y_data']
        print(f"Data loaded from {filepath}")
        return X_data, Y_data
    else:
        print(f"File {filepath} not found")
        return None, None

# Sequence data
def sequence(X_train, Y_train, window):
    X_seq, y_seq = [], []
    for i in range(len(X_train) - window):
        X_seq.append(X_train[i:i+window])
        y_seq.append(Y_train[i+window])
    return np.array(X_seq), np.array(y_seq)

# Load data
X_data_train, Y_data_train = load_data('Training_Data_S&P500_3.npz')
X_data_test, Y_data_test = load_data('Test_Data_S&P500_3.npz')

# Select features
X_data_train = X_data_train[:,5:13]
X_data_test = X_data_test[:,5:13]

# Sequence data
X_train, Y_train = sequence(X_data_train, Y_data_train, 50)
X_test, Y_test = sequence(X_data_test, Y_data_test, 50)

# Neural network model
def cnn(X_train, X_test, Y_train, Y_test):
    model = keras.models.Sequential()
    model.add(keras.layers.Conv1D(128, kernel_size=(3), activation='relu', input_shape=X_train.shape[1:]))
    model.add(keras.layers.MaxPooling1D(pool_size=(2)))
    model.add(keras.layers.Conv1D(256, kernel_size=(3), activation='relu'))
    model.add(keras.layers.MaxPooling1D(pool_size=(2)))
    model.add(keras.layers.LSTM(1024))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    optimizer = keras.optimizers.Adam(learning_rate=1e-5)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    history = model.fit(X_train, Y_train, validation_data = (X_test,Y_test), epochs=30, batch_size=1024)
    model.summary()
    plt.figure(figsize=(12,6))
    plt.semilogy(history.history['accuracy']) 
    plt.semilogy(history.history['val_accuracy'])
    plt.title('Training Performance')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    return model

# Train model
#model = cnn(X_train, X_test, Y_train, Y_test)

# Function to calculate accuracy
def accuracy(model, data, labels):
    X = data[:,5:13]
    Y = labels
    features, outcomes = sequence(X, Y, 50)
    pred = model.predict(features)
    positions = np.zeros(len(pred))
    for i in range(0, len(pred)):
        if pred[i] > 0.50:
            positions[i] = 1
        elif pred[i] < 0.50:
            positions[i] = 0
    n = 0
    for i in range(0, len(positions)):
        if positions[i] == outcomes[i]:
            n += 1
    accuracy = (n/len(positions))*100
    print(Y)
    print(positions)
    return accuracy

# Load model 
model = keras.models.load_model("positions_model_S&P500_3.keras")
data, labels = load_data('Test_Data_S&P500_3.npz')

# Evaluate accuracy
accuracy = accuracy(model, data, labels)
print(accuracy)

# Save model
#model.save('positions_model_S&P500_3.keras')