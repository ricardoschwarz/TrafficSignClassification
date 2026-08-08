import itertools

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix


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


def plot_history(history, title="Titel"):
	"""plot loss and accuracy from history"""
	plt.title=title
	fig, ax = plt.subplots(2, 1)
	fig.suptitle(title)
	ax[0].plot(history.history['loss'], color='b', label="Training loss")
	ax[0].plot(history.history['val_loss'], color='r', label="validation loss")
	ax[0].set_xlabel('Epochs')
	ax[0].set_ylabel('Loss')
	legend = ax[0].legend(loc='best', shadow=True)

	ax[1].plot(history.history['accuracy'], color='b', label="Training accuracy")
	ax[1].plot(history.history['val_accuracy'], color='r', label="Validation accuracy")
	ax[1].set_xlabel('Epochs')
	ax[1].set_ylabel('Validation')
	legend = ax[1].legend(loc='best', shadow=True)
	plt.show(block=False)


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


def compute_confusion_matrix(model, dataset):
	"""Computes the confusion matrix for a model against a batched (images, labels) dataset"""
	y_pred = model.predict(dataset)
	y_pred_classes = np.argmax(y_pred, axis=1)
	y_true = np.concatenate([labels.numpy() for _, labels in dataset], axis=0)

	confusion_mtx = confusion_matrix(y_true, y_pred_classes)
	return confusion_mtx
