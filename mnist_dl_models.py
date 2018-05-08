import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D, Dropout


def load_model(i, filepath):

    model = Sequential()
    if i == 0:
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(200, activation='relu'))
        model.add(Dropout(.5))
        model.add(Dense(200, activation='relu'))
        model.add(Dropout(.5))
        model.add(Dense(10, activation='softmax'))

    elif i == 1:
        model.add(Conv2D(64, (4, 4), activation='relu', input_shape=(28, 28, 1)))
        model.add(Conv2D(64, (4, 4), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(100, activation='relu'))
        model.add(Dropout(.5))
        model.add(Dense(100, activation='relu'))
        model.add(Dropout(.5))
        model.add(Dense(10, activation='softmax'))

    elif i == 2:
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(50, activation='relu'))
        model.add(Dropout(.5))
        model.add(Dense(50, activation='relu'))
        model.add(Dropout(.5))
        model.add(Dense(10, activation='softmax'))

    elif i == 3:
        model.add(Conv2D(16, kernel_size=(2, 2), activation='relu', input_shape=(28, 28, 1)))
        model.add(Conv2D(16, (2, 2), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(50, activation='relu'))
        model.add(Dropout(.5))
        model.add(Dense(50, activation='relu'))
        model.add(Dropout(.5))
        model.add(Dense(50, activation='relu'))
        model.add(Dropout(.5))
        model.add(Dense(10, activation='softmax'))

    elif i == 4:
        model.add(Conv2D(16, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
        model.add(Conv2D(16, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(.5))
        model.add(Flatten())
        model.add(Dense(50, activation='relu'))
        model.add(Dropout(.5))
        model.add(Dense(10, activation='softmax'))

    model.load_weights(filepath)
    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.SGD(lr=0.01), metrics=['accuracy'])
    return model
