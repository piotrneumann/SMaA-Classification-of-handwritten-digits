import pickle as pkl
import warnings
import time

from predict import predict

TEST_DATA = pkl.load(open('train.pkl', mode='rb'))

def error(y_pred, y_true):
    return sum(y_pred != y_true)/len(y_pred)

def predict_test():

    start_time = time.time()
    y_pred = predict(TEST_DATA[0][29000:])
    t = TEST_DATA[0][29000:]
    print(y_pred)
    print("--- %s seconds ---" % (time.time() - start_time))
    print("Blad wynosi:", error(y_pred, TEST_DATA[1][29000:]))

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    # img = Image.open('ja.jpg')
    # cover = resizeimage.resize_cover(img, [200, 200])
    # # cover.save('test-image-cover.jpeg', img.format)
    # plt.imshow(cover)
    # plt.show()

    # for i in range(0, 10):
    #     img = np.reshape(TEST_DATA[0][i],(26,26))
    #     plt.imshow(img)
    #     plt.show()
    print("Start")
    predict_test()