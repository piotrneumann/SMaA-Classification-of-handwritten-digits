import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow_core.python.keras.layers.core import Dense

# from KNN import evaluateKNN

print(tf.__version__)

# Now we can load the MNIST dataset using the Keras helper function.
mnist = tf.keras.datasets.mnist
np_utils = tf.keras.utils
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# print(x_train[0])
#
print(y_train.shape)
print(x_train.shape)

plt.imshow(x_train[1],cmap=plt.cm.binary)
plt.show()

x_train = tf.keras.utils.normalize(x_train, axis=1)  # scales data between 0 and 1
x_test = tf.keras.utils.normalize(x_test, axis=1)  # scales data between 0 and 1
print()
model = tf.keras.models.Sequential()  # a basic feed-forward model
model.add(tf.keras.layers.Flatten())  # takes our 28x28 and makes it 1x784
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))  # a simple fully-connected layer, 128 units, relu activation
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))  # a simple fully-connected layer, 128 units, relu activation
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))  # our output layer. 10 units for 10 classes. Softmax for probability distribution

model.compile(optimizer='adam',  # Good default optimizer to start with
              loss='sparse_categorical_crossentropy',  # how will we calculate our "error." Neural network aims to minimize loss.
              metrics=['accuracy'])  # what to track
batch_size = 1000
num_epoch = 10
model.fit(x_train, y_train, epochs=num_epoch)  # train the model

val_loss, val_acc = model.evaluate(x_test, y_test)  # evaluate the out of sample data with model
print(val_loss)  # model's loss (error)
print(val_acc)  # model's accuracy
tmp_image = x_train[1].reshape(1, -1)
pred = model.predict(tmp_image)
print(pred)
print(np.argmax(pred[0]))