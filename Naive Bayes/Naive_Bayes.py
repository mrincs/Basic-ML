import numpy as np
import pandas as pd

class NaiveBayes:

    def __init__(self):
        self._eps = 1e-6

    def _set_variables(self, X, y):
        self.X = X
        self.y = y
        self.classes = np.unique(y) # Assume all classes are present
        self.params = [None] * len(self.classes)



    # P_X_given_y
    def _compute_gaussian_likelihood(self, mu, sigma, x, offset=True):
        if offset:
            return 1.0/(sigma * np.sqrt(2 * np.pi) + self._eps) * np.exp(-(x - mu)**2 / (2 * sigma**2 + self._eps))
        else:
            return 1.0/(sigma * np.sqrt(2 * np.pi)) * np.exp(-(x - mu)**2 / (2 * sigma**2))


    # P_y_given_X
    def _compute_posteriors(self, x):

        posteriors = [None] * len(self.classes)
        for i, cls in enumerate(self.classes):

            prior = np.mean(self.y == cls)
            conditional_likelihood = 1
            for feature, param in zip(x, self.params[i]):
                conditional_likelihood = conditional_likelihood * self._compute_gaussian_likelihood(param['mu'], param['sigma'], feature)
            posteriors[i] = prior * conditional_likelihood
        return posteriors


    def fit(self, X, y):
        self._set_variables(X, y)

        for i, cls in enumerate(self.classes):
            class_indices = np.where(self.y == cls)
            select_X = X[class_indices].T

            param = []
            for feature in select_X:
                param.append({'mu': np.mean(feature), 'sigma': np.std(feature)})
            self.params[i] = param



    def predict(self, X):
        y_predict = [None] * len(X)
        for i, x in enumerate(X):
            y_predict[i] = np.argmax(self._compute_posteriors(x))
        return y_predict





def _test_Naive_Bayes():

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

    nb = NaiveBayes()
    nb.fit(X_tr, y_tr)
    y_predict = nb.predict(X_te)
    print('Predictions: ', y_predict)


if __name__ == '__main__':
    _test_Naive_Bayes()
