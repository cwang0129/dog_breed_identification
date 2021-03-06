import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from data_manager import RESIZE_DIMENSION, int2label, label2int, load_data
NUM_CLASS = len(int2label)
INPUT_DIM = RESIZE_DIMENSION * RESIZE_DIMENSION * 3

#x_train = np.random.random((1000, 20))
x_train, y_train = load_data("data/train", keep_original=True)
#y_train = keras.utils.to_categorical(np.random.randint(10, size=(1000, 1)), num_classes=10)
y_train = keras.utils.to_categorical(y_train, num_classes=NUM_CLASS)
#x_test = np.random.random((100, 20))
x_test, y_test = load_data("data/train", keep_original=True)
y_test = keras.utils.to_categorical(y_test, num_classes=NUM_CLASS)

model = Sequential()
# input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
# this applies 32 convolution filters of size 3x3 each.
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(RESIZE_DIMENSION, RESIZE_DIMENSION, 3)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(NUM_CLASS, activation='softmax'))

sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
#model.compile(loss='categorical_crossentropy', optimizer=sgd)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])
model.fit(x_train, y_train,
          epochs=20,
          batch_size=128)
model.save("cnn_dog_sp_2.model")
score = model.evaluate(x_test, y_test, batch_size=128)
print score