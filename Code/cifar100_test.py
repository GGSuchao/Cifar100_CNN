from __future__ import print_function
import keras
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score
from keras.datasets import cifar100
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import keras.backend as K
import os

num_classes = 100

# Network Parameters
batch_size = 32

(X_train, y_train), (X_test, y_test) = cifar100.load_data(label_mode = 'fine')

# Input Normalization
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = load_model('./save_models/keras_cifar100_trained_model.h5')

scores = model.evaluate(X_test, y_test, batch_size = batch_size)
print('Test loss: ', scores[0])
print('Test accuracy: ', scores[1])