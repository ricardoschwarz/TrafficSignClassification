# ----------
# script to train a simple cnn (later maybe AlexNet) with the German Trafic Sign Dataset from https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/published-archive.html
#
# run image_preprocessing.py once before running this script
# ----------

import os

import tensorflow as tf
from matplotlib import pyplot as plt

from data import load_training_data
from labels import classes
from models import get_basic_model, get_complex_model
from plotting import compute_confusion_matrix, plot_confusion_matrix, plot_history, show_images

MAX_EPOCHS = 100
MODELS_DIR = "models"

# load the dataset
train_image_dir = "data/train"
train_ds, val_ds, test_ds = load_training_data(train_image_dir)


def callbacks_for(name):
	os.makedirs(MODELS_DIR, exist_ok=True)
	return [
		tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
		tf.keras.callbacks.ModelCheckpoint(
			filepath=os.path.join(MODELS_DIR, f"{name}_best.weights.h5"),
			monitor="val_loss",
			save_best_only=True,
			save_weights_only=True,
		),
	]


# base model
basic_model = get_basic_model()
history = basic_model.fit(train_ds, epochs=MAX_EPOCHS, validation_data=val_ds, callbacks=callbacks_for("basic_model"))
plot_history(history, "Basic Model")
confusion_mtx_basic = compute_confusion_matrix(basic_model, val_ds)
plot_confusion_matrix(confusion_mtx_basic, classes = range(9), title="CMatrix - Basic Model")

# more complex model
complex_model = get_complex_model()
history = complex_model.fit(train_ds, epochs=MAX_EPOCHS, validation_data=val_ds, callbacks=callbacks_for("complex_model"))
plot_history(history, "Complex Model")
confusion_mtx_complex = compute_confusion_matrix(complex_model, val_ds)
plot_confusion_matrix(confusion_mtx_complex, classes = range(9), title="CMatrix - Complex Model")
complex_model.save(os.path.join(MODELS_DIR, "complex_model.keras"))

# test models
test_loss_basic, test_acc_basic = basic_model.evaluate(test_ds)
print("---Basic Model Test\nTest Loss: {0}\nTest Accuracy: {1}".format(test_acc_basic, test_acc_basic))
test_loss_complex, test_acc_complex = complex_model.evaluate(test_ds)
print("---Complex Model Test\nTest Loss: {0}\nTest Accuracy: {1}".format(test_acc_complex, test_acc_complex))

sample_images, sample_labels = next(iter(train_ds))
show_images(sample_images.numpy(), sample_labels.numpy(), classes)

plt.tight_layout() # for beautiful plots
plt.show() # to pause execution
