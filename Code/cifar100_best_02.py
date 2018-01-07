from __future__ import print_function
import keras
from keras.datasets import cifar100
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import BatchNormalization
from keras.optimizers import SGD, Adam
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from keras import regularizers
from keras import backend as K
from matplotlib import pyplot
import numpy as np
import os

K.set_image_dim_ordering('tf')

num_classes = 100
epochs = 250
batch_size = 128
learning_rate = 0.1

save_dir = os.path.join(os.getcwd(), 'save_models')
model_name = 'keras_cifar100_trained_model_best_02.h5'

# load data
(X_train, y_train), (X_test, y_test) = cifar100.load_data(label_mode = 'fine')

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

mean = np.mean(X_train, axis = (0, 1, 2, 3))
std = np.std(X_train, axis = (0, 1, 2, 3))
X_train = (X_train - mean) / (std + 1e-7)
X_test = (X_test - mean) / (std + 1e-7)

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

datagen = ImageDataGenerator(rotation_range = 15, width_shift_range = 0.1, height_shift_range = 0.1, horizontal_flip = True)


model = Sequential()
model.add(Conv2D(64, (3, 3), padding = 'same', input_shape = (32, 32, 3), activation = 'relu', kernel_regularizer = regularizers.l2(0.0005)))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Conv2D(64, (3, 3), padding = 'same', activation = 'relu', kernel_regularizer = regularizers.l2(0.0005)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(128, (3, 3), padding = 'same', activation = 'relu', kernel_regularizer = regularizers.l2(0.0005)))
model.add(BatchNormalization())
model.add(Dropout(0.4))
model.add(Conv2D(128, (3, 3), padding = 'same', activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(256, (3, 3), padding = 'same', activation = 'relu', kernel_regularizer = regularizers.l2(0.0005)))
model.add(BatchNormalization())
model.add(Dropout(0.4))
model.add(Conv2D(256, (3, 3), padding = 'same', activation = 'relu', kernel_regularizer = regularizers.l2(0.0005)))
model.add(BatchNormalization())
model.add(Dropout(0.4))
model.add(Conv2D(256, (3, 3), padding = 'same', activation = 'relu', kernel_regularizer = regularizers.l2(0.0005)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(512, (3, 3), padding = 'same', activation = 'relu', kernel_regularizer = regularizers.l2(0.0005)))
model.add(BatchNormalization())
model.add(Dropout(0.4))
model.add(Conv2D(512, (3, 3), padding = 'same', activation = 'relu', kernel_regularizer = regularizers.l2(0.0005)))
model.add(BatchNormalization())
model.add(Dropout(0.4))
model.add(Conv2D(512, (3, 3), padding = 'same', activation = 'relu', kernel_regularizer = regularizers.l2(0.0005)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(512, activation = 'relu', kernel_regularizer = regularizers.l2(0.0005)))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(num_classes, activation = 'softmax'))

sgd = SGD(lr = learning_rate, momentum = 0.9, decay = 1e-6, nesterov = True)
model.compile(loss = 'categorical_crossentropy', optimizer = sgd, metrics = ['accuracy'])
print(model.summary())

for i in range(1, epochs):
    if i % 25 == 0 and i > 0:
    	learning_rate /= 2
        sgd = SGD(lr = learning_rate, momentum = 0.9, decay = 1e-6, nesterov = True)
        model.compile(loss = 'categorical_crossentropy', optimizer = sgd, metrics = ['accuracy'])
    
    model.fit_generator(datagen.flow(X_train, y_train, batch_size = batch_size), steps_per_epoch = X_train.shape[0] // batch_size, epochs = i, validation_data = (X_test, y_test), initial_epoch = i - 1, workers = 4)

if not os.path.isdir(save_dir):
	os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)

scores = model.evaluate(X_test, y_test, batch_size = batch_size)
print('Test loss: ', scores[0])
print('Test accuracy: ', scores[1])

pyplot.figure(figsize = (15, 8))
pyplot.plot(Hist.history['acc'])
pyplot.plot(Hist.history['val_acc'])
pyplot.title('Accuracy', size = 20)
pyplot.ylabel('accuracy', size = 20)
pyplot.xlabel('epoch', size = 20)
pyplot.legend(['train', 'test'], loc = 'upper left', prop = {'size':15})
pyplot.savefig('Best_02_acc.png')

pyplot.figure(figsize = (15, 8))
pyplot.plot(Hist.history['loss'])
pyplot.plot(Hist.history['val_loss'])
pyplot.title('Loss', size = 20)
pyplot.ylabel('loss', size = 20)
pyplot.xlabel('epoch', size = 20)
pyplot.legend(['train', 'test'], loc = 'upper left', prop = {'size':15})
pyplot.savefig('Best_02_los.png')