import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow_core.python.keras.layers.core import Dense

from KNN import evaluateKNN

print(tf.__version__)

# Now we can load the MNIST dataset using the Keras helper function.
mnist = tf.keras.datasets.mnist
np_utils = tf.keras.utils
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(y_test[0])

# evaluateKNN(x_train, x_test)

plt.imshow(x_train[0], cmap=plt.cm.binary)
# plt.imshow(x_train[1], cmap=plt.cm.binary)
# plt.imshow(x_test[0], cmap=plt.cm.binary)
plt.interactive(False)
plt.show()
# x_train = tf.keras.utils.normalize(x_train, axis=1)
# x_test = tf.keras.utils.normalize(x_test, axis=1)
#
# model = tf.keras.models.Sequential()
# model.add(tf.keras.layers.Flatten())
# model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
# model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
# model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))
# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])
# model.fit(x_train, y_train, epochs=3)
# val_loss, val_acc = model.evaluate(x_test, y_test)
# print(val_loss)
# print(val_acc)
# flatten 28*28 images to a 784 vector for each image
# num_pixels = x_train.shape[1] * x_train.shape[2]
# x_train = x_train.reshape((x_train.shape[0], num_pixels)).astype('float32')
# x_test = x_test.reshape((x_test.shape[0], num_pixels)).astype('float32')
# x_train = x_train / 255
# x_test = x_test / 255
#
# y_train = np_utils.to_categorical(y_train)
# y_test = np_utils.to_categorical(y_test)
# num_classes = y_test.shape[1]

# plt.imshow(x_train[0],cmap=plt.cm.binary)
# plt.imshow(x_train[1],cmap=plt.cm.binary)
# plt.imshow(x_train[2],cmap=plt.cm.binary)
# plt.show()

# def baseline_model():
# 	# create model
# 	model = tf.keras.models.Sequential
# 	model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))
# 	model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))
# 	# Compile model
# 	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# 	return model

# build the model
# model = baseline_model()
# Fit the model
# model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200, verbose=2)
# # Final evaluation of the model
# scores = model.evaluate(X_test, y_test, verbose=0)
# print("Baseline Error: %.2f%%" % (100-scores[1]*100))
