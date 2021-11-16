from data import X, y, symbols, inverse_map

from tensorflow import keras
from tensorflow.keras import layers

def Network():
    image = layers.Input((1, 50, 200))

    # pain and suffering
    out = layers.Conv2D(16, (3, 3), activation="relu", padding="same")(image)
    out = layers.MaxPooling2D((2, 2), padding="same")(out)
    out = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(out)
    out = layers.MaxPooling2D((2, 1), padding="same")(out)
    out = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(out)
    out = layers.MaxPooling2D((2, 5), padding="same")(out)
    out = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(out)
    """
    out = layers.MaxPooling2D((2, 1), padding="same")(out)
    out = layers.Conv2D(256, (3, 3), activation="relu", padding="same")(out)
    out = layers.MaxPooling2D((2, 1), padding="same")(out)
    out = layers.Conv2D(256, (3, 3), activation="relu", padding="same")(out)
    """
    out = layers.BatchNormalization()(out)

    # squeeze into bidirectional lstm lol
    out = keras.backend.squeeze(out, axis=1)
    out = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(out)
    out = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(out)
    out = keras.backend.expand_dims(out, axis=2)
    out = layers.Conv2D(len(symbols), (2, 2), activation="relu", padding="same")(out)
    out = keras.backend.squeeze(out, axis=2)
    out = layers.Softmax()(out)

    # model
    model = keras.Model(image, out)
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics="accuracy")

    return model

model = Network()
model.summary()

# decide on how this is going to work
X_train, X_test, y_train, y_test = X[200:], X[:-200], y[200:], y[:-200]

model.fit(X_train, y_train, epochs=100)
print("\nscore: ", model.evaluate(X_test, y_test))

v = model.predict(X_test[0][None, None, :, :])
print(inverse_map(v))
