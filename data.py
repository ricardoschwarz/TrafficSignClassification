import os

import numpy as np
import tensorflow as tf
from PIL import Image

IMAGE_HEIGHT = 50
IMAGE_WIDTH = 50
CHANNELS = 3


def _index_files(image_dir):
    """Walks image_dir (one subfolder per class) and returns (file_paths, labels).

    Labels are assigned by sorted subfolder name, matching how the dataset's
    class-index subfolders (e.g. "00000", "00001", ...) already line up with
    the label ids in labels.py.
    """
    class_names = sorted(
        name for name in os.listdir(image_dir)
        if os.path.isdir(os.path.join(image_dir, name))
    )
    class_to_index = {name: index for index, name in enumerate(class_names)}

    file_paths = []
    labels = []
    for subdir, _dirs, files in os.walk(image_dir):
        class_name = os.path.basename(subdir)
        if class_name not in class_to_index:
            continue
        for file in files:
            if file.endswith(".ppm"):
                file_paths.append(os.path.join(subdir, file))
                labels.append(class_to_index[class_name])

    return file_paths, labels


def _load_image(path, label):
    """Decodes a .ppm file via PIL (tf.io/tf.image have no native PPM support)."""
    def _load(path_tensor):
        image = Image.open(path_tensor.numpy().decode("utf-8")).convert("RGB")
        return np.asarray(image, dtype=np.float32) / 255.0

    image = tf.py_function(_load, [path], tf.float32)
    image.set_shape((IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS))
    return image, label


def load_training_data(image_dir, batch_size=32, test_split=0.2, val_split=0.2, seed=15):
    """Loads training images as tf.data.Dataset pipelines, split into train/val/test.

    Mirrors the historical split behavior: `test_split` is carved off first, then
    `val_split` is carved off the remainder, leaving the rest for training.

    Parameters:
    image_dir(str): directory of training images, laid out as one subfolder per class
        (see README), already resized to 50x50x3 by image_preprocessing.py
    batch_size(int): batch size for the returned datasets
    test_split(float): fraction of all images held out for the test set
    val_split(float): fraction of the remaining (non-test) images held out for validation
    seed(int): shuffle seed, for reproducible splits

    Return:
    (train_ds, val_ds, test_ds): batched, prefetched tf.data.Dataset objects of
        (image, label) pairs, images shaped (50, 50, 3) and normalized to [0, 1],
        labels as int class indices (0-42)
    """
    file_paths, labels = _index_files(image_dir)
    total = len(file_paths)
    print("Working with {0} training images".format(total))

    full_ds = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    full_ds = full_ds.shuffle(buffer_size=total, seed=seed, reshuffle_each_iteration=False)

    test_size = int(total * test_split)
    val_size = int((total - test_size) * val_split)
    train_size = total - test_size - val_size
    print("Train set size: {0}, Val set size: {1}, Test set size: {2}".format(
        train_size, val_size, test_size))

    test_ds = full_ds.take(test_size)
    train_val_ds = full_ds.skip(test_size)
    val_ds = train_val_ds.take(val_size)
    train_ds = train_val_ds.skip(val_size)

    autotune = tf.data.AUTOTUNE
    train_ds = (train_ds.map(_load_image, num_parallel_calls=autotune)
                .batch(batch_size).prefetch(autotune))
    val_ds = (val_ds.map(_load_image, num_parallel_calls=autotune)
                .batch(batch_size).prefetch(autotune))
    test_ds = (test_ds.map(_load_image, num_parallel_calls=autotune)
                .batch(batch_size).prefetch(autotune))

    return train_ds, val_ds, test_ds
