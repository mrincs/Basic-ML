import numpy as np
import pandas as pd

class KNN:

    def __init__(self, k):
        self.k = k

    def _set_variables(self, X, y):
        self.X = X
        self.y = y

    def fit(self, X, y):
        self._set_variables(X, y)
        return

    def predict(self, X):
        y_predict = [None] * len(X)
        for i, x in enumerate(X):
            distances = [np.linalg.norm(x-x_tr) for x_tr in self.X]
            top_k_indices = np.argsort(distances)[:self.k]
            top_k_labels = self.y[top_k_indices]
            y_predict[i] = np.argmax(np.bincount(top_k_labels))

        return y_predict



def _test_KNN():

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

    knn_1 = KNN(3)
    knn_1.fit(X_tr, y_tr)
    y_predict = knn_1.predict(X_te)
    print('Predictions: ', y_predict)


    knn_2 = KNN(5)
    knn_2.fit(X_tr, y_tr)
    y_predict = knn_2.predict(X_te)
    print('Predictions: ', y_predict)


if __name__ == '__main__':
    _test_KNN()
