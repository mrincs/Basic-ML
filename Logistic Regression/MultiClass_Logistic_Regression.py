import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split



np.random.seed(4312)

class LogisticRegression:

    def __init__(self):
        self.n_iters = 100
        self.lr = 1e-2

    def _set_variables(self, X, y):
        self.X = X
        self.classes = np.unique(y)
        self.n_classes = len(self.classes)
        self.y = np.eye(self.n_classes)[y]

    def _init_params(self):
        mu = 0
        sigma = 0.1
        self.params = [np.random.normal(mu, sigma, self.X.shape[1]) for _ in range(self.n_classes)]

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def fit(self, X, y):
        self._set_variables(X, y)
        self._init_params()
        for iter in range(self.n_iters):
            for c in range(self.n_classes):
                y_predict = np.array(list(map(self._sigmoid, X.dot(self.params[c]))))
                self.params[c] = self.params[c] - self.lr * X.T.dot(y_predict-self.y[:,c])


    def predict(self, X):
        y_predict_per_class = [-1] * self.n_classes
        for c in range(self.n_classes):
            y_predict_per_class[c] = np.array(list(map(self._sigmoid, X.dot(self.params[c]))))
        print(np.array(y_predict_per_class).shape)
        y_predict = np.argmax(y_predict_per_class, axis=0)
        print(y_predict.shape)
        return y_predict


    def accuracy(self, y, y_predict):
        print('y: ',y)
        print('y_pred: ', y_predict)
        return np.sum(y == y_predict) / len(y)



def _test_Logistic_Regression():


    digits = load_digits()
    print("Image Data Shape: ",digits.data.shape)
    print("Label Data Shape: ",digits.target.shape)

    X_tr, X_te, y_tr, y_te = train_test_split(digits.data, digits.target, test_size=0.2, random_state=0)

    logreg = LogisticRegression()
    logreg.fit(X_tr, y_tr)
    y_predict = logreg.predict(X_te)
    print('Predictions: ', y_predict)

    print('Accuracy: ', logreg.accuracy(y_te, y_predict)*100)



if __name__ == '__main__':
    _test_Logistic_Regression()
