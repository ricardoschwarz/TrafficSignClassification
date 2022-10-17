# class labels
classes = {
    0: "Speedlimit = 20",
    1: "Speedlimit = 30",
    2: "Speedlimit = 50",
    3: "Speedlimit = 60",
    4: "Speedlimit = 70",
    5: "Speedlimit = 80",
    6: "End of Speedlimit = 80",
    7: "End of Speedlimit = 100",
    8: "Speedlimit = 120",
    9: "No overtaking allowed for cars",
    10: "No overtaking allowed for trucks",
    11: "Right of way at next junction",
    12: "Right of way on this road",
    13: "Yield right of way",
    14: "Stop. Yield right of way",
    15: "All vehicles banned for this road",
    16: "All trucks banned for this road",
    17: "No Entry!",
    18: "Danger Spot",
    19: "Sharp Left Corner",
    20: "Sharp Right Corner",
    21: "Double Curve",
    22: "Uneven Road",
    23: "Slip Hazard",
    24: "Road narrows",
    25: "Roadworks",
    26: "Traffic Light",
    27: "Pedestrians",
    28: "Children",
    29: "Cyclists",
    30: "Slipperiness",
    31: "Deer Path",
    32: "End of all constraints",
    33: "Prescribed driving direction right",
    34: "Prescribed driving direction left",
    35: "Prescribed driving direction ahead",
    36: "Prescribed driving direction ahead and right",
    37: "Prescribed driving direction ahead and left",
    38: "Prescribed passing at the right side",
    39: "Prescribed passing at the left side",
    40: "Roundabout",
    41: "Overtaking is now allowed",
    42: "Overtaking is now allowed for trucks",
}


import matplotlib.pyplot as plt
def show_images(images, labels, classes):
	"""shows a diagram from input images with corresponding class labels

	Parameters:
	images(PIL image): images to display
	labels(str): array of labels
	classes(str): array of all possible classes

	"""    
	plt.figure(figsize=(10,10))
	for i in range(25):
		plt.subplot(5,5,i+1)
		plt.imshow(images[i], cmap="binary")
		plt.grid(False)
		plt.xticks([])
		plt.yticks([])
		plt.xlabel(classes[labels[i]])
					
	plt.show(block=False)


def load_training_data(image_dir):
	"""Loads training images.
	IMPORTANT: make sure all images are resized properly (50x50x3)

	Parameters:
	image_dir(str): directory of training images

	Return:
	(dataset, y_train): tuple of dataset (shape: number_img, image_width, image_height, img_depth) and corresponding labels
	"""
	train_files = []
	y_train = []

	import os
	for subdir, dirs, files in os.walk(image_dir):
		for file in files: 
			if file.endswith(".ppm"):
				file_name = os.path.join(subdir, file)
				train_files.append(file_name)
				y_train.append(int(file_name[11:16]))

	print("Working with {0} training images".format(len(train_files)))

	from scipy import ndimage
	from keras.utils import img_to_array, load_img

	# Original Dimensions
	image_height = 50
	image_width = 50
	channels = 3

	import numpy as np
	dataset = np.ndarray(shape=(len(train_files), image_height, image_width, channels),
						dtype=np.float32)

	i = 0
	for _file in train_files:
		img = load_img(_file)  # this is a PIL image
		# Convert to Numpy Array
		x = img_to_array(img)
		# Normalize
		x = x / 255.0
		dataset[i] = x
		i += 1
		if i % 1000 == 0:
			print("%d images to array" % i)
	print("All images to array!")

	return (dataset, y_train)
  

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
	fig = plt.figure()
	fig.suptitle(title)
	plt.imshow(cm, interpolation='nearest', cmap=cmap)
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

	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	plt.show(block=False)


from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
def get_basic_model(input_shape=(50, 50, 3), output_shape=43):
	"""Create and compile a basic model"""
	model = Sequential([
		Flatten(input_shape=input_shape),
		Dense(128, activation="relu"),
		Dense(output_shape, activation="softmax")
	])
	model.compile(optimizer='adam',
		loss='sparse_categorical_crossentropy',
		metrics=['accuracy'])
	return model


def get_complex_model(input_shape=(50, 50, 3), output_shape=43):
	"""Create and compile a more complex model with a convolutional layer + pooling layer"""
	modelConv = Sequential()
	modelConv.add(Conv2D(10, (3,3), strides= (1,1), padding='same', input_shape=input_shape, activation='relu'))
	modelConv.add(MaxPooling2D(pool_size=(2, 2)))
	modelConv.add(Flatten())
	modelConv.add(Dense(100, activation="relu"))
	modelConv.add(Dense(output_shape, activation="softmax"))

	modelConv.compile(optimizer='adam',
					loss='sparse_categorical_crossentropy',
					metrics=['accuracy'])

	return modelConv

def plot_history(history, title="Titel"):
	"""plot loss and accuracy from history"""
	import matplotlib.pyplot as plt
	plt.title=title
	fig, ax = plt.subplots(2, 1)
	fig.suptitle(title)
	ax[0].plot(history.history['loss'], color='b', label="Training loss")
	ax[0].plot(history.history['val_loss'], color='r', label="validation loss", axes=ax[0])
	ax[0].set_xlabel('Epochs')
	ax[0].set_ylabel('Loss')
	legend = ax[0].legend(loc='best', shadow=True)

	ax[1].plot(history.history['accuracy'], color='b', label="Training accuracy")
	ax[1].plot(history.history['val_accuracy'], color='r', label="Validation accuracy")
	ax[1].set_xlabel('Epochs')
	ax[1].set_ylabel('Validation')
	legend = ax[1].legend(loc='best', shadow=True)
	plt.show(block=False)

import numpy as np
def compute_confusion_matrix(model, X_val, y_val):
	"""plot the confusion matrix"""
	y_pred = model.predict(X_val)

	# Convert predictions classes to one hot vectors 
	y_pred_classes = np.argmax(y_pred,axis = 1) 
	
	confusion_mtx = confusion_matrix(y_val, y_pred_classes)
	return confusion_mtx