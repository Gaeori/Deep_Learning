from keras.datasets import mnist
import numpy as np

num_feature = 28 * 28
learning_rate = 1e-4
training_step = 10


def pre_processing_data():
    (train_data, train_label), (test_data, test_label) = mnist.load_data()

    (train_data, train_label) = map(list, zip(*[(x, y) for x, y in zip(train_data, train_label) if y == 0 or y == 1]))
    (test_data, test_label) = map(list, zip(*[(x, y) for x, y in zip(test_data, test_label) if y == 0 or y == 1]))

    train_data, train_label = np.array(train_data, np.float32), np.array(train_label, np.float32)
    test_data, test_label = np.array(test_data, np.float32), np.array(test_label, np.float32)

    train_data, test_data = train_data.reshape([-1, num_feature]), test_data.reshape([-1, num_feature])
    train_data, test_data = train_data / 255., test_data / 255.

    return train_data, train_label, test_data, test_label


def logistic_regression(x, w, b):
    return 1. / (1. + np.exp(-np.dot(w, x) - b))


def train(train_data, train_label):
    w = np.random.uniform(-1, 1, num_feature)
    b = np.random.uniform(-1, 1)

    for step in range(training_step):
        dw = np.zeros(num_feature, dtype='float32')
        db = 0.
        y_hat = []
        loss = 0.

        for x, y in zip(train_data, train_label):
            a = logistic_regression(x, w, b)
            y_hat.append(a)
            loss += y * np.log(a) + (1 - y) * np.log(1-a)

            dw += (y - a) * x
            db += (y - a)

        print(f"Epoch {step+1} / {training_step}, train_accuracy: {accuracy(y_hat, train_label)}, train_loss: {-(loss/len(train_data))}")

        w += learning_rate * dw
        b += learning_rate * db

    return w, b


def accuracy(y_predict, y_true):
    correct_prediction = np.equal(np.round(y_predict), y_true.astype(np.int64))
    return np.mean(correct_prediction.astype(np.float32))


def logistic_regression_wo_vectorization(test_data, w, b):
    predict = []
    for t in test_data:
        predict.append(logistic_regression(t, w, b))
    return predict


def main():
    train_data, train_label, test_data, test_label = pre_processing_data()
    w, b = train(train_data, train_label)

    prediction = logistic_regression_wo_vectorization(test_data, w, b)

    print(f"Test Accuracy: {accuracy(prediction, test_label)}")


if __name__ == "__main__":
    main()
