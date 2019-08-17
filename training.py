# ----------
# script to train a simple cnn (later maybe AlexNet) with the German Trafic Sign Dataset from https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/published-archive.html
# ----------

# useful methods

import numpy as np
import matplotlib.pyplot as plt
def show_images(images, labels, classes):
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i+1)
        plt.imshow(images[i], cmap="binary")
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        plt.xlabel(classes[labels[i]])
        plt.show()


# Look at confusion matrix
from sklearn.metrics import confusion_matrix
import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


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

# Import keras libraries
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
#from keras.optimizers import RMSprop
#from keras.callbacks import ReduceLROnPlateau

# Build a baseline model
model = Sequential()
model.add(Flatten(input_shape=(50,50,3)))
model.add(Dense(255, activation="relu"))
model.add(Dense(11, activation="softmax"))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=5, validation_data=(X_val, y_val))

# Plot the loss and accuracy curves for training and validation of the baseline model
from matplotlib import pyplot as plt
fig, ax = plt.subplots(2,1)
ax[0].plot(history.history['loss'], color='b', label="Training loss")
ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['acc'], color='b', label="Training accuracy")
ax[1].plot(history.history['val_acc'], color='r',label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)
plt.show() 

# test_loss, test_acc = model.evaluate(X_test, y_test)

# Building a more complex model with a convolutional layer + pooling layer
modelConv = Sequential()
modelConv.add(Conv2D(10, (3,3), strides= (1,1), padding='same', input_shape=(50,50,3), activation='relu'))
modelConv.add(MaxPooling2D(pool_size=(2, 2)))
modelConv.add(Flatten())
modelConv.add(Dense(100, activation="relu"))
modelConv.add(Dense(11, activation="softmax"))

modelConv.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

history = modelConv.fit(X_train, y_train, validation_data = (X_val, y_val) ,epochs=10)

# Plot loss and accuracy curves for second model
fig, ax = plt.subplots(2, 1)
ax[0].plot(history.history['loss'], color='b', label="Training loss")
ax[0].plot(history.history['val_loss'], color='r',
           label="validation loss", axes=ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['acc'], color='b', label="Training accuracy")
ax[1].plot(history.history['val_acc'], color='r', label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)
plt.show()

# Predict the values from the validation dataset
y_pred = model.predict(X_val)
# Convert predictions classes to one hot vectors 
y_pred_classes = np.argmax(y_pred,axis = 1) 
# Convert validation observations to one hot vectors
y_true = y_val 
# compute the confusion matrix
confusion_mtx = confusion_matrix(y_true, y_pred_classes)

# plot the confusion matrix
plot_confusion_matrix(confusion_mtx, classes = range(11))
plt.show()

classes = {
    0: "Limit 20",
    1: "Limit 30",
    2: "Limit 50",
    3: "Limit 60",
    4: "Limit 70",
    5: "Limit 80",
    6: "Limit Not 80",
    7: "Limit 100",
    8: "Limit 120"
}

show_images(X_train, y_train, classes)
