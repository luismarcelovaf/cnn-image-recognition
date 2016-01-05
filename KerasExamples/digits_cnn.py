import numpy as np
from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils

digits = load_digits()
X = (digits.data / 16.).astype(np.float32)
y = np_utils.to_categorical(digits.target)

X_train, X_test, y_train, y_test = train_test_split(X, y)

X_train = X_train.reshape(X_train.shape[0], 1, 8, 8) # 8x8 images, grayscale
X_test = X_test.reshape(X_test.shape[0], 1, 8, 8) # 8x8 images, grayscale

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

model = Sequential()

model.add(Convolution2D(32, 1, 3, 3, border_mode='full'))
model.add(Activation('relu'))
model.add(Convolution2D(32, 32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(poolsize=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, 128))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(128, 10))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adadelta')

batch_size = 64

model.fit(X_train, y_train, batch_size=batch_size,
          nb_epoch=20, show_accuracy=True,
          verbose=1, validation_data=(X_test, y_test))

score = model.evaluate(X_test, y_test, show_accuracy=True, verbose=0)
print('Test accuracy:', score[1])
