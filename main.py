import os
import cv2
import numpy as np

from keras import layers
from keras.models import Model
from keras.utils.vis_utils import plot_model

dataset_path = 'input/dataset'
valid_characters = "0123456789abcdefghijklmnopqrstuvwxyz"


def preprocessing():
    num_samples = len(os.listdir(dataset_path))
    images = np.zeros((num_samples, 50, 200, 1))
    keys = np.zeros((5, num_samples, len(valid_characters)))

    for i, img_path in enumerate(os.listdir(dataset_path)):
        img = cv2.imread(os.path.join(dataset_path, img_path), cv2.IMREAD_GRAYSCALE)
        img = img / 255.0
        img = np.reshape(img, (50, 200, 1))
        targs = np.zeros((5, len(valid_characters)))
        for j, char in enumerate(img_path[:5]):
            index = valid_characters.find(char)
            targs[j, index] = 1
        images[i] = img
        keys[:, i] = targs
    return images, keys


def create_model():
    input_img = layers.Input(shape=img_shape)
    conv1 = layers.Conv2D(16, (3, 3), padding='same', activation='relu')(input_img)
    mp1 = layers.MaxPooling2D(padding='same')(conv1)
    conv2 = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(mp1)
    mp2 = layers.MaxPooling2D(padding='same')(conv2)
    conv3 = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(mp2)
    mp3 = layers.MaxPooling2D(padding='same')(conv3)
    conv4 = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(mp3)
    bn1 = layers.BatchNormalization()(conv4)
    mp4 = layers.MaxPooling2D(padding='same')(bn1)

    flat = layers.Flatten()(mp4)
    outputs = []
    for _ in range(5):
        dense = layers.Dense(64, activation='relu')(flat)
        dropout = layers.Dropout(0.5)(dense)
        result = layers.Dense(len(valid_characters), activation='sigmoid')(dropout)
        outputs.append(result)

    model = Model(input_img, outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


x, y = preprocessing()
img_shape = x[0].shape

model = create_model()
model.summary()
plot_model(model, to_file='model_plot.png', show_shapes=False, show_layer_names=False)

model.fit(x, [y[0], y[1], y[2], y[3], y[4]], batch_size=32, epochs=30, verbose=1, validation_split=.2)

score= model.evaluate(x,[y[0], y[1], y[2], y[3], y[4]],verbose=1)
print('Test Loss and accuracy:', score)
model.save('trained_model')
