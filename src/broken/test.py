import os
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

# load in our pre trained model
model = tf.keras.models.load_model("./ocr_model")
model.summary()

batch_size = 20
img_width = 200
img_height = 50
downsample_factor = 4

# this is absolutely hardcoded to fit to our dataset don't worry about it
# this is super good for the demo lol
data_dir = Path("./data/")
images = sorted(list(map(str, list(data_dir.glob("*.png")))))
labels = [img.split(os.path.sep)[-1].split(".png")[0] for img in images]
symbols = set(char for label in labels for char in label)

# max length of any given CAPTCHA image
max_length = max([len(label) for label in labels])

# Mapping symbols to integers
char_to_num = layers.StringLookup(
    vocabulary=list(symbols), mask_token=None
)

# Mapping integers back to original symbols
num_to_char = layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
)

def encode_single_sample(img_path, label):
    # 1. Read image
    img = tf.io.read_file(img_path)
    # 2. Decode and convert to grayscale
    img = tf.io.decode_png(img, channels=1)
    # 3. Convert to float32 in [0, 1] range
    img = tf.image.convert_image_dtype(img, tf.float32)
    # 4. Resize to the desired size
    img = tf.image.resize(img, [img_height, img_width])
    # 5. Transpose the image because we want the time
    # dimension to correspond to the width of the image.
    img = tf.transpose(img, perm=[1, 0, 2])
    # 6. Map the symbols in label to numbers
    label = char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))
    # 7. Return a dict as our model is expecting two inputs
    return {"image": img, "label": label}

x = tf.data.Dataset.from_tensor_slices((["./data/226md.png"], ["226md"]))

x = (
    x.map(
        encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE
    )
    .batch(1)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)
print(x)

def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
        :, :max_length
    ]
    # Iterate over the results and get back the text
    output_text = []
    for res in results:
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        output_text.append(res)
    return output_text

print(decode_batch_predictions(model.predict(x)))

for batch in x.take(1):
    img = (batch["image"][0, :, :, 0] * 255).numpy().astype(np.uint8)
    img = img.T
    plt.imshow(img)
    plt.savefig("out.png")
