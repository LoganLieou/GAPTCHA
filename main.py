import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import os

valid_characters = "0123456789abcdefghijklmnopqrstuvwxyz"


def preprocessing():
    n_samples = len(os.listdir('input/dataset'))
    X = np.zeros((n_samples, 50, 200, 1))  # 1070*50*200
    y = np.zeros((5, n_samples, len(valid_characters)))  # 5*1070*36

    for i, pic in enumerate(os.listdir('input/dataset')):
        # Read image as grayscale
        img = cv2.imread(os.path.join('input/dataset', pic), cv2.IMREAD_GRAYSCALE)
        pic_target = pic[:-4]
        if len(pic_target) < 6:
            # Scale and reshape image
            img = img / 255.0
            img = np.reshape(img, (50, 200, 1))
            # Define targets and code them using OneHotEncoding
            targs = np.zeros((5, len(valid_characters)))
            for j, l in enumerate(pic_target):
                ind = valid_characters.find(l)
                targs[j, ind] = 1
            X[i] = img
            y[:, i] = targs

    # Return final data
    return X, y


X, y = preprocessing()
X_train, y_train = X[:970], y[:, :970]
X_test, y_test = X[970:], y[:, 970:]
img_shape = X[0].shape


from keras import layers
from keras.models import Model
from keras.models import load_model
from keras import callbacks


def create_model():
    img = layers.Input(shape=img_shape)  # Get image as an input and process it through some Convs
    conv1 = layers.Conv2D(16, (3, 3), padding='same', activation='relu')(img)
    mp1 = layers.MaxPooling2D(padding='same')(conv1)  # 100x25
    conv2 = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(mp1)
    mp2 = layers.MaxPooling2D(padding='same')(conv2)  # 50x13
    conv3 = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(mp2)
    bn = layers.BatchNormalization()(conv3)
    mp3 = layers.MaxPooling2D(padding='same')(bn)  # 25x7

    # Get flattened vector and make 5 branches from it. Each branch will predict one letter
    flat = layers.Flatten()(mp3)
    outs = []
    for _ in range(5):
        dens1 = layers.Dense(64, activation='relu')(flat)
        drop = layers.Dropout(0.5)(dens1)
        res = layers.Dense(len(valid_characters), activation='sigmoid')(drop)

        outs.append(res)

    # Compile model and return it
    model = Model(img, outs)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
    return model


model=create_model()
model.summary()

hist = model.fit(X_train, [y_train[0], y_train[1], y_train[2], y_train[3], y_train[4]], batch_size=32, epochs=30,verbose=1, validation_split=0.2)


# Define function to predict captcha
def predict(filepath):
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        img = img / 255.0
    else:
        print("Not detected");
    res = np.array(model.predict(img[np.newaxis, :, :, np.newaxis]))
    ans = np.reshape(res, (5, 36))
    l_ind = []
    probs = []
    for a in ans:
        l_ind.append(np.argmax(a))
        #probs.append(np.max(a))

    capt = ''
    for l in l_ind:
        capt += valid_characters[l]
    return capt#, sum(probs) / 5

score= model.evaluate(X_test,[y_test[0], y_test[1], y_test[2], y_test[3], y_test[4]],verbose=1)
print('Test Loss and accuracy:', score)

# Check model on some samples
model.evaluate(X_test, [y_test[0], y_test[1], y_test[2], y_test[3], y_test[4]])
print(predict('input/2en7g.png'))
print(predict('input/3mxdn.png'))
print(predict('input/gecmf.png'))
print(predict('input/8n5p3.png'))