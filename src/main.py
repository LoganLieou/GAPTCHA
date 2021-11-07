from data import X, Y, symbols

from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.InputLayer((50, 200, 1)),
    layers.Conv2D(16, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2), padding="same"),
    layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2), padding="same"),
    layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2), padding="same"),
    layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 1), padding="same"),
    layers.Conv2D(256, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 1), padding="same"),
    layers.BatchNormalization(),
    layers.LSTM(256, return_sequences=True),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(len(symbols), activation="relu"),
    layers.Softmax()
])

model.summary()
model.fit(X, Y, epochs=50)
