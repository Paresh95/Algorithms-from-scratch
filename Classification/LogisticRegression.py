import numpy as np

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class LogisticRegression:
    def __init__(self, learning_rate=1e-3, n_steps=100, l2_penalty=True, lambda_val=0.1, verbose=True):
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.l2_penalty = l2_penalty
        self.lambda_val = lambda_val
        self.verbose = verbose

    def add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)

    def sigmoid(self, z):
        return 1/(1 + np.exp(-z))

    def loss(self, h, y):
        return (-y * np.log(h) - (1-y) * np.log(1-h)).mean()

    def fit(self, X, y):

        X = self.add_intercept(X)

        # initialise weights
        self.W = np.random.rand(X.shape[1])

        for i in range(self.n_steps):
            z = np.dot(X, self.W)
            h = self.sigmoid(z)
            if self.l2_penalty:
                DW = (np.dot(X.T, (h-y)) / y.size) + (2 * self.lambda_val * self.W)
            else:
                DW = np.dot(X.T, (h-y)) / y.size
            self.W -= self.learning_rate * DW

            if self.verbose is True and i % 1000 == 0:
                z = np.dot(X, self.W)
                h = self.sigmoid(z)
                print(f'loss: {self.loss(h, y)} \t')

    def predict_prob(self, X):

        X = self.add_intercept(X)

        return self.sigmoid(np.dot(X, self.W))

    def predict(self, X, threshold):
        boolean_prediction = self.predict_prob(X) >= threshold
        return boolean_prediction.astype(int)


if __name__ == "__main__":
    # make data
    data = datasets.make_classification(n_samples=10000, n_features=10, n_classes=2)
    X = data[0]
    y = data[1]

    # split into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75,
                                                        random_state=1)

    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)

    # instantiate object and fit model
    model = LogisticRegression(learning_rate=1e-3, n_steps=10000, l2_penalty=True, lambda_val=0.1, verbose=True)
    model.fit(X_train, y_train)

    # training loss
    y_pred = model.predict(X_train, threshold=0.5)
    y_pred_prob = model.predict_prob(X_train)
    print(y_pred)
    print(y_pred_prob)
    print(accuracy_score(y_train, y_pred))

    # testing loss
    y_pred = model.predict(X_test, threshold=0.5)
    y_pred_prob = model.predict_prob(X_test)
    print(y_pred)
    print(y_pred_prob)
    print(accuracy_score(y_test, y_pred))




