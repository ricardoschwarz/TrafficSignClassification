from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential


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
