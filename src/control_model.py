from tensorflow import keras
import numpy as np
import cv2

valid_characters = "0123456789abcdefghijklmnopqrstuvwxyz"

model = keras.models.load_model('trained_model')

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


print(predict('input/3mxdn.png'))
print(predict('input/4gb3f.png'))
print(predict('input/5expp.png'))
print(predict('input/cgcgb.png'))
print(predict('input/geyn5.jpg'))