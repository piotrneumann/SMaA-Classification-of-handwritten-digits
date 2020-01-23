# K-Nearest Neighbor Classification

from __future__ import print_function
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn import datasets
from skimage import exposure
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# import imutils
# import cv2

mnist = tf.keras.datasets.mnist
np_utils = tf.keras.utils
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(y_train[0])

# load the MNIST digits dataset
mnist = datasets.load_digits()
#
#
# # print (mnist.data)
#
# # Training and testing split,
# # 75% for training and 25% for testing
(trainData, testData, trainLabels, testLabels) = train_test_split(np.array(mnist.data), mnist.target, test_size=0.25,
                                                                  random_state=42)

#
# # take 10% of the training data and use that for validation
(trainData, valData, trainLabels, valLabels) = train_test_split(trainData, trainLabels, test_size=0.1, random_state=84)

t1 = trainData[0]
B = np.reshape(t1, (8, 8))

t2 = trainData[1]
B1 = np.reshape(t2, (8, 8))

t3 = trainData[2]
B2 = np.reshape(t3, (8, 8))

t4 = trainData[3]
B3 = np.reshape(t4, (8, 8))

print(trainData)
print("shape " + str(trainData.shape))
print("shape " + str(trainData[0].shape))

print(trainData.shape)
print(trainLabels)

# mnist = tf.keras.datasets.mnist
# np_utils = tf.keras.utils
# (trainData, valData), (trainLabels, valLabels) = mnist.load_data()
#
# print(trainData.shape)
# print(trainLabels.shape)

# plt.imshow(trainData[0],cmap=plt.cm.binary)
# plt.show()

# Checking sizes of each data split
print("training data points: {}".format(len(trainLabels)))
print("validation data points: {}".format(len(valLabels)))
# print("testing data points: {}".format(len(testLabels)))

# initialize the values of k for our k-Nearest Neighbor classifier along with the
# list of accuracies for each value of k
kVals = range(1, 30, 2)
accuracies = []

# loop over kVals
for k in range(1, 30, 2):
    # train the classifier with the current value of `k`
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(trainData, trainLabels)

    # evaluate the model and print the accuracies list
    score = model.score(valData, valLabels)
    print("k=%d, accuracy=%.2f%%" % (k, score * 100))
    accuracies.append(score)

# largest accuracy
# np.argmax returns the indices of the maximum values along an axis
i = np.argmax(accuracies)
print("k=%d achieved highest accuracy of %.2f%% on validation data" % (kVals[i],
                                                                       accuracies[i] * 100))

# Now that I know the best value of k, re-train the classifier
model = KNeighborsClassifier(n_neighbors=kVals[i])
model.fit(trainData, trainLabels)

# Predict labels for the test set
predictions = model.predict(valData)

image = valData[i]
tmp_image = image.reshape(1, -1)

Bi = np.reshape(tmp_image, (8, 8))

prediction = model.predict(tmp_image)[0]
prediction1 = model.predict(tmp_image)
print(prediction)
plt.imshow(tmp_image,cmap=plt.cm.binary)
plt.show()

# Evaluate performance of model for each of the digits
print("EVALUATION ON TESTING DATA")
# print(classification_report(testLabels, predictions))

# some indices are classified correctly 100% of the time (precision = 1)
# high accuracy (98%)

# check predictions against images
# loop over a few random digits
# for i in np.random.randint(0, high=len(testLabels), size=(5,)):
#     # np.random.randint(low, high=None, size=None, dtype='l')
#     image = testData[i]
#     prediction = model.predict(image)[0]
#
#     # convert the image for a 64-dim array to an 8 x 8 image compatible with OpenCV,
#     # then resize it to 32 x 32 pixels for better visualization
#     image = image.reshape((8, 8)).astype("uint8")
#     image = exposure.rescale_intensity(image, out_range=(0, 255))
#     image = imutils.resize(image, width=32, inter=cv2.INTER_CUBIC)
#
#     # show the prediction
#     print("I think that digit is: {}".format(prediction))
#     cv2.imshow("Image", image)
#     cv2.waitKey(0) # press enter to view each one!
