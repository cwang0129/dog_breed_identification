import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from data_manager import RESIZE_DIMENSION, int2label, label2int, load_data
NUM_CLASS = len(int2label)
INPUT_DIM = RESIZE_DIMENSION * RESIZE_DIMENSION * 3

#x_train = np.random.random((1000, 20))
x_train, y_train = load_data("data/train", keep_original=False)
#y_train = keras.utils.to_categorical(np.random.randint(10, size=(1000, 1)), num_classes=10)
y_train = keras.utils.to_categorical(y_train, num_classes=NUM_CLASS)
#x_test = np.random.random((100, 20))
x_test, y_test = load_data("data/train", keep_original=False)
y_test = keras.utils.to_categorical(y_test, num_classes=NUM_CLASS)

model = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 20-dimensional vectors.
model.add(Dense(10240, activation='relu', input_dim=INPUT_DIM))
model.add(Dropout(0.5))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(NUM_CLASS, activation='softmax'))

sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

model.fit(x_train, y_train,
          epochs=10,
          batch_size=128)

model.save("mlp_dog_sp.model")
score = model.evaluate(x_test, y_test, batch_size=128)
print score