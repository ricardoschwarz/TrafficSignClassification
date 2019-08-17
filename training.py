# ----------
# script to train a cnn (probably AlexNet) with the German Trafic Sign Dataset from https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/published-archive.html
# ----------


# load the dataset
import os
train_folder = "data/train"

filenames = []
for subdir, dirs, files in os.walk(train_folder):
    for file in files: 
        if file.endswith(".ppm"):
            filenames.append(os.path.join(subdir, file))

print("Working with {0} images".format(len(filenames)))

from scipy import ndimage
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

train_files = []
y_train = []
i=0
for _file in filenames:
    train_files.append(_file)
    label_in_file = _file.find("_")
    y_train.append(int(_file[11:16]))
    
print("Files in train_files: %d" % len(train_files))

# Original Dimensions
image_height = 50
image_width = 50

channels = 3
nb_classes = 1

import numpy as np
dataset = np.ndarray(shape=(len(train_files), image_height, image_width, channels),
                     dtype=np.float32)

import PIL
i = 0
for _file in train_files:
    img = load_img(_file)  # this is a PIL image
    img = img.resize((image_height,image_width), PIL.Image.ANTIALIAS)
    # Convert to Numpy Array
    x = img_to_array(img)  
    # x = x.reshape((30, 29, 3))
    # Normalize
    x = x / 255.0
    dataset[i] = x
    i += 1
    if i % 250 == 0:
        print("%d images to array" % i)
print("All images to array!")

from sklearn.model_selection import train_test_split

#Splitting 
X_train, X_val, y_train, y_val = train_test_split(dataset, y_train, test_size=0.2, random_state=33)
print("Train set size: {0}, Val set size: {1}, Test set size: NOT YET".format(len(X_train), len(X_val)))

# from tensorflow.python import keras
# model = keras.Sequential([
#         keras.layers.Flatten(input_shape=(28, 28)),
#         keras.layers.Dense(128, activation="relu"),
#         keras.layers.Dense(10, activation="softmax")
#     ])

# model.compile(optimizer='adam',
#         loss='sparse_categorical_crossentropy',
#         metrics=['accuracy'])

# model.fit(X_train, y_train, epochs=5)

# test_loss, test_acc = model.evaluate(X_test, y_test)