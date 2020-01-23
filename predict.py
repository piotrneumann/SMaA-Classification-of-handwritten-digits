import pickle as pkl
import numpy as np

TEST_DATA = pkl.load(open('train.pkl', mode='rb'))
def predict(x):

    dane1 = TEST_DATA[0][:2000]
    dane2 = TEST_DATA[0][3000:4000]
    dane3 = TEST_DATA[0][5000:6000]
    dane4 = TEST_DATA[0][7000:8000]
    dane5 = TEST_DATA[0][9000:10000]
    dane6 = TEST_DATA[0][11000:12000]
    dane7 = TEST_DATA[0][13000:14000]
    dane8 = TEST_DATA[0][15000:15100]
    x_train = np.concatenate([dane1,dane2,dane3,dane4,dane5,dane6,dane7])

    dane1 = TEST_DATA[1][:2000]
    dane2 = TEST_DATA[1][3000:4000]
    dane3 = TEST_DATA[1][5000:6000]
    dane4 = TEST_DATA[1][7000:8000]
    dane5 = TEST_DATA[1][9000:10000]
    dane6 = TEST_DATA[1][11000:12000]
    dane7 = TEST_DATA[1][13000:14000]
    dane8 = TEST_DATA[1][15000:15100]
    y_train = np.concatenate([dane1,dane2,dane3,dane4,dane5,dane6,dane7])

    prediction = np.zeros(shape=(len(x[:,0]), 1))

    hamm_dist = hamming_distance(x, x_train)
    # print(hamm_dist)
    leng = len(hamm_dist[:,0])
    for i in range(0, leng):
        prediction[i,0] = y_train[np.argmin(hamm_dist[i,:])]
    return prediction


def my_predict(x_train, y_train, x_test, y_test):
    print(x_train[0])
    pass

def hamming_distance(X, X_train):

    N1 = len(X[:,0])
    N2 = len(X_train[:,0])
    distance_matrix = np.zeros(shape=(N1, N2))

    for i in range(0, N1):
        for j in range(0, N2):
            distance_matrix[i, j] = np.count_nonzero(X[i, :] != X_train[j, :])
    return distance_matrix