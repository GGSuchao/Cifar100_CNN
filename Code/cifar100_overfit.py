from __future__ import print_function
import keras
import tensorflow as tf
from keras.datasets import cifar100
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers
from matplotlib import pyplot
import keras.backend as K
import os

num_classes = 100

# Network Parameters
batch_size = 32
epochs = 100
save_dir = os.path.join(os.getcwd(), 'save_models')
model_name = 'keras_cifar100_trained_model_overfit.h5'

(X_train, y_train), (X_test, y_test) = cifar100.load_data(label_mode = 'fine')

# Input Normalization
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Define the Model
model = Sequential()
# Convolutional layer 1, 3*3 filter * 32
model.add(Conv2D(32, (3, 3), padding = 'same', input_shape = X_train.shape[1:], kernel_regularizer = regularizers.l2(0.0005)))
model.add(Activation('relu'))

# Convolutional layer 2, 3*3 filter * 32
model.add(Conv2D(32, (3, 3), kernel_regularizer = regularizers.l2(0.0005)))
model.add(Activation('relu'))

# Pooling layer 3, MaxPooling with 2 * 2
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.25))

# Convolutional layer 4, 3*3 filter * 64
model.add(Conv2D(64, (3, 3), padding = 'same', kernel_regularizer = regularizers.l2(0.0005)))
model.add(Activation('relu'))

# Convolutional layer 5, 3*3 filter * 64
model.add(Conv2D(64, (3, 3), kernel_regularizer = regularizers.l2(0.0005)))
model.add(Activation('relu'))

# Pooling layer 6, MaxPooling with 2 * 2
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())

# Fully connected layer 7, 512 units
model.add(Dense(512, kernel_regularizer = regularizers.l2(0.0005)))
model.add(Activation('relu'))
model.add(Dropout(0.5))

# Fully connected layer 8, 100 units
model.add(Dense(num_classes))
model.add(Activation('softmax'))

# Compile the model
rmsprop = keras.optimizers.RMSprop(lr = 0.0001, decay = 1e-6)
model.compile(optimizer = rmsprop, loss = 'categorical_crossentropy', metrics = ['accuracy'])

datagen = ImageDataGenerator(rotation_range = 15, width_shift_range = 0.1, height_shift_range = 0.1, horizontal_flip = True)

Hist = model.fit_generator(datagen.flow(X_train, y_train, batch_size = batch_size), epochs = epochs, validation_data = (X_test, y_test), workers = 4)

# Save Model and Weights
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
pyplot.savefig('overfit_acc.png')

pyplot.figure(figsize = (15, 8))
pyplot.plot(Hist.history['loss'])
pyplot.plot(Hist.history['val_loss'])
pyplot.title('Loss', size = 20)
pyplot.ylabel('loss', size = 20)
pyplot.xlabel('epoch', size = 20)
pyplot.legend(['train', 'test'], loc = 'upper left', prop = {'size':15})
pyplot.savefig('overfit_los.png')
