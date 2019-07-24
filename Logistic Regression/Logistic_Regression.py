import numpy as np
import pandas as pd



np.random.seed(4312)

class LogisticRegression:

    def __init__(self):
        self.n_iters = 10000
        self.lr = 1e-2

    def _set_variables(self, X, y):
        self.X = X
        self.y = y

    def _init_params(self):
        mu = 0
        sigma = 0.1
        self.params = np.random.normal(mu, sigma, self.X.shape[1])
        print(self.params.shape)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def fit(self, X, y):
        self._set_variables(X, y)
        self._init_params()
        for iter in range(self.n_iters):
            y_predict = np.array(list(map(self._sigmoid, X.dot(self.params))))
            # print(y_predict, (y_predict-self.y).shape)
            self.params = self.params - self.lr * X.T.dot(y_predict-self.y)


    def predict(self, X):
        y_predict = list(map(self._sigmoid, X.dot(self.params)))
        y_predict = np.round(y_predict)
        return y_predict


    def accuracy(self, y, y_predict):
        print('y: ',y)
        print('y_pred: ', y_predict)
        return np.sum(y == y_predict) / len(y)



def _test_Logistic_Regression():

    # Create our feature variables
    height = [6,5.92,5.58,5.92,5,5.5,5.42,5.75]
    weight = [180,190,170,165,100,150,130,150]
    foot_size = [12,11,12,10,6,8,7,9]
    # Create our target variable
    gender = ['male','male','male','male','female','female','female','female']


    # Create an empty dataframe
    df = pd.DataFrame({
        'height': height,
        'weight': weight,
        'foot_size': foot_size
        })
    print(df)





    # Prepare data
    gender_id = np.array([0 if i == 'male' else 1 for i in gender])
    X_tr = df.values[:6]
    y_tr = gender_id[:6]
    X_te = df.values[6:]
    y_te = gender_id[6:]


    logreg = LogisticRegression()
    logreg.fit(X_tr, y_tr)
    y_predict = logreg.predict(X_te)
    print('Predictions: ', y_predict)

    print('Accuracy: ', logreg.accuracy(y_te, y_predict)*100)



if __name__ == '__main__':
    _test_Logistic_Regression()
