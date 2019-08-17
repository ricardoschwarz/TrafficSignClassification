# ----------
# script to train a simple cnn (later maybe AlexNet) with the German Trafic Sign Dataset from https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/published-archive.html
#
# run image_preprocessing.py once before running this script
# ----------

## useful methods

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
					
	plt.show()

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
	from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

	# Original Dimensions
	image_height = 50
	image_width = 50
	channels = 3

	import numpy as np
	dataset = np.ndarray(shape=(len(train_files), image_height, image_width, channels),
						dtype=np.float32)

	import PIL
	i = 0
	for _file in train_files:
		img = load_img(_file)  # this is a PIL image
		# Convert to Numpy Array
		x = img_to_array(img)  
		# Normalize
		x = x / 255.0
		dataset[i] = x
		i += 1
		if i % 250 == 0:
			print("%d images to array" % i)
	print("All images to array!")

	return (dataset, y_train)
  

## main

if __name__ == "__main__":

	# load the dataset
	train_image_dir = "data/train"
	X_data, y_data = load_training_data(train_image_dir)

	#Splitting 
	from sklearn.model_selection import train_test_split
	X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=15)
	X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=33)
	print("Train set size: {0}, Val set size: {1}, Test set size: {2}".format(len(X_train), len(X_val), len(X_test)))

	from tensorflow.python import keras
	model = keras.Sequential([
			keras.layers.Flatten(input_shape=(50, 50, 3)),
			keras.layers.Dense(128, activation="relu"),
			keras.layers.Dense(10, activation="softmax")
		])

	model.compile(optimizer='adam',
			loss='sparse_categorical_crossentropy',
			metrics=['accuracy'])

	model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

	test_loss, test_acc = model.evaluate(X_test, y_test)

	classes = { 
		0 : "Limit 20", 
		1 : "Limit 30", 
		2 : "Limit 50",
		3 : "Limit 60",
		4 : "Limit 70",
		5 : "Limit 80",
		6 : "Limit Not 80",
		7 : "Limit 100",
		8 : "Limit 120"
	}
 
	show_images(X_train, y_train, classes)
