import numpy as np
import keras
from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense, Activation
from keras.layers.normalization import BatchNormalization
from data_manager import RESIZE_DIMENSION, int2label, label2int, load_data
NUM_CLASS = len(int2label)
INPUT_DIM = RESIZE_DIMENSION * RESIZE_DIMENSION * 3

x_train, y_train = load_data("data/train", keep_original=True)
y_train = keras.utils.to_categorical(y_train, num_classes=NUM_CLASS)
x_test, y_test = load_data("data/train", keep_original=True)
y_test = keras.utils.to_categorical(y_test, num_classes=NUM_CLASS)

model = Sequential()
model.add(BatchNormalization(input_shape=(RESIZE_DIMENSION, RESIZE_DIMENSION, 3)))
model.add(Conv2D(filters=16, kernel_size=3, kernel_initializer='he_normal', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(BatchNormalization())

model.add(Conv2D(filters=32, kernel_size=3, kernel_initializer='he_normal', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(BatchNormalization())

model.add(Conv2D(filters=64, kernel_size=3, kernel_initializer='he_normal', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(BatchNormalization())

model.add(Conv2D(filters=128, kernel_size=3, kernel_initializer='he_normal', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(BatchNormalization())

model.add(Conv2D(filters=256, kernel_size=3, kernel_initializer='he_normal', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(BatchNormalization())

model.add(GlobalAveragePooling2D())

model.add(Dense(NUM_CLASS, activation='softmax'))


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train,
          epochs=100,
          batch_size=128)
model.save("net_dog_sp100.model")
score = model.evaluate(x_test, y_test, batch_size=128)
print score
