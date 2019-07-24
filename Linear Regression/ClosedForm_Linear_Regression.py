import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.cross_validation import train_test_split


np.random.seed(4312)

class LinearRegression:

    def __init__(self):
        pass

    def _set_variables(self, X, y):
        self.X = X
        self.y = y


    # Closed form solution
    def fit(self, X, y):
        self._set_variables(X, y)

        inv_X_T_dot_X = np.linalg.inv(X.T.dot(X))
        self.params = inv_X_T_dot_X.dot(X.T.dot(y))


    def predict(self, X):
        y_predict = X.dot(self.params)
        print(y_predict)
        return y_predict


    def mse(self, y, y_predict):
        print('y: ',y)
        print('y_pred: ', y_predict)
        return np.linalg.norm(y-y_predict) / len(y)



def _test_Linear_Regression():

    boston_dataset = load_boston()

    X = boston_dataset['data']
    y = boston_dataset['target']

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0)

    print(X_tr.shape, y_tr.shape, X_te.shape, y_te.shape)

    linreg = LinearRegression()
    linreg.fit(X_tr, y_tr)
    y_predict = linreg.predict(X_te)

    print('Mean Square Error: ', linreg.mse(y_te, y_predict))



if __name__ == '__main__':
    _test_Linear_Regression()
