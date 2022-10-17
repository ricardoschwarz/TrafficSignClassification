# ----------
# script to train a simple cnn (later maybe AlexNet) with the German Trafic Sign Dataset from https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/published-archive.html
#
# run image_preprocessing.py once before running this script
# ----------

from utils import load_training_data, get_basic_model, get_complex_model, compute_confusion_matrix, plot_confusion_matrix, plot_history, show_images, classes
from matplotlib import pyplot as plt
import numpy as np

# load the dataset
train_image_dir = "data/train"
X_data, y_data = load_training_data(train_image_dir)

# splitting 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=15)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=33)
print("Train set size: {0}, Val set size: {1}, Test set size: {2}".format(len(X_train), len(X_val), len(X_test)))

X_train = np.array(X_train)
y_train = np.array(y_train)
X_val = np.array(X_val)
y_val = np.array(y_val)
X_test = np.array(X_test)
y_test = np.array(y_test)

# base model
basic_model = get_basic_model()
history = basic_model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))
plot_history(history, "Basic Model")
confusion_mtx_basic = compute_confusion_matrix(basic_model, X_val, y_val)
plot_confusion_matrix(confusion_mtx_basic, classes = range(9), title="CMatrix - Basic Model")

# more complex model
complex_model = get_complex_model()
history = complex_model.fit(X_train, y_train, epochs=10, validation_data = (X_val, y_val))
plot_history(history, "Complex Model")
confusion_mtx_complex = compute_confusion_matrix(complex_model, X_val, y_val)
plot_confusion_matrix(confusion_mtx_complex, classes = range(9), title="CMatrix - Complex Model")
complex_model.save('models/complex_model')

# test models
test_loss_basic, test_acc_basic = basic_model.evaluate(X_test, y_test)
print("---Basic Model Test\nTest Loss: {0}\nTest Accuracy: {1}".format(test_acc_basic, test_acc_basic))
test_loss_complex, test_acc_complex = complex_model.evaluate(X_test, y_test)
print("---Complex Model Test\nTest Loss: {0}\nTest Accuracy: {1}".format(test_acc_complex, test_acc_complex))

show_images(X_train, y_train, classes)

plt.tight_layout() # for beautiful plots
plt.show() # to pause execution
