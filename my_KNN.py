import tensorflow as tf
from sklearn.neighbors import KNeighborsClassifier
import time

mnist = tf.keras.datasets.mnist
np_utils = tf.keras.utils
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(y_train[0])
accuracies = []
# model = KNeighborsClassifier(n_neighbors=2)

print("shape: " + str(x_train.shape))
print("shape: " + str(y_train.shape))

te = x_train[0].reshape(-1)

rows, cols = (7000, 784)

train_data = [[0 for i in range(cols)] for j in range(rows)]
for i in range(rows):
    train_data[i] = (x_train[i].reshape(-1))
train_label = y_train[:rows]

validate_data = [[0 for i in range(cols)] for j in range(rows)]
for i in range(rows):
    validate_data[i] = (x_test[i].reshape(-1))
validate_label = y_test[:rows]
# ------------------------------------------------------------------------
# FOR TEST K
# for k in range(1, 21, 2):
#     # train the classifier with the current value of `k`
#     model = KNeighborsClassifier(n_neighbors=k)
#     model.fit(train_data, train_label)
#
#     # evaluate the model and print the accuracies list
#     start_time = time.time()
#     score = model.score(validate_data, validate_label)
#     print("--- %s seconds ---" % (time.time() - start_time))
#     print("k=%d, accuracy=%.2f%%" % (k, score * 100))
#     accuracies.append(score)
# -------------------------------------------------------------------------


# train the classifier with the current value of `k`
model = KNeighborsClassifier(n_neighbors=1)
model.fit(train_data, train_label)

# evaluate the model and print the accuracies list
start_time = time.time()
score = model.score(validate_data, validate_label)
print("--- %s seconds ---" % (time.time() - start_time))
print("k=%d, accuracy=%.2f%%" % (1, score * 100))
accuracies.append(score)


# t1 = x_train[0]
# t2 = t1.reshape(-1)
# model.fit(train_data, train_label)
# score = model.score(validate_data, validate_label)
# print("accuracy=%.2f%%" % (score * 100))
# predictions = model.predict(validate_data)
# print(predictions)

# evaluate the model and print the accuracies list
# score = model.score(x_test, y_test)
# print("k=%d, accuracy=%.2f%%" % (1, score * 100))
# accuracies.append(score)