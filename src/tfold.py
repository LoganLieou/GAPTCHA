from data import X, y, symbols

from tensorflow import keras
from tensorflow.keras import layers

def Network():
    image = layers.Input((50, 200, 1))

    out = layers.Conv2D(16, (3, 3), activation="relu", padding="same")(image)
    out = layers.MaxPooling2D((2, 2), padding="same")(out)
    out = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(out)
    out = layers.MaxPooling2D((2, 2), padding="same")(out)
    out = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(out)
    out = layers.MaxPooling2D((2, 2), padding="same")(out)
    out = layers.BatchNormalization()(out)
    out = layers.MaxPooling2D((2, 2), padding="same")(out)

    """
    out = keras.backend.expand_dims(out, axis=2)
    out = layers.Conv2D(len(symbols), (2, 2), activation="relu", padding="same")(out)
    """

    out = keras.backend.squeeze(out, axis=0)
    out = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(out)
    out = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(out)
    out = keras.backend.expand_dims(out, axis=2)
    out = layers.Conv2D(len(symbols), (2, 2), activation="relu", padding="same")(out)
    out = keras.backend.squeeze(out, axis=2)
    out = layers.Softmax()(out)

    model = keras.Model(image, out)
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics="accuracy")

    return model

model = Network()
model.summary()
# model.fit(X, y, epochs=30)

