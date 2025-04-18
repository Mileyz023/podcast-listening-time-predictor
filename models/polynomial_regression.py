import numpy as np
import itertools
from models.linear_regression_model import LinearRegression

class PolynomialRegression(LinearRegression):
    def __init__(self, degree, learning_rate, num_iterations):
        self.degree = degree
        super().__init__(learning_rate, num_iterations)

    def transform(self, X):
        if X.ndim == 1:
            X = X[:, np.newaxis]
        num_examples, num_features = X.shape
        features = [np.ones((num_examples, 1))]  # bias term

        for j in range(1, self.degree + 1):
            for combination in itertools.combinations_with_replacement(range(num_features), j):
                feature = np.ones(num_examples)
                for index in combination:
                    feature *= X[:, index]
                features.append(feature[:, np.newaxis])

        return np.concatenate(features, axis=1)

    def fit(self, X, Y):
        self.X = X
        self.Y = Y

        X_poly = self.transform(X)
        X_poly_normalized = X_poly.copy()

        if X_poly.shape[1] > 1:
            X_poly_normalized[:, 1:] = self.normalize(X_poly_normalized[:, 1:])

        num_examples, num_features = X_poly_normalized.shape
        self.W = np.zeros((num_features, 1))

        for _ in range(self.num_iterations):
            Y_pred = X_poly_normalized.dot(self.W)
            dW = -(2 / num_examples) * X_poly_normalized.T.dot(self.Y - Y_pred)
            self.W -= self.learning_rate * dW
            cost = np.sqrt(np.mean((self.Y - Y_pred) ** 2))
            self.cost_history.append(cost)

        return self

    def predict(self, X):
        X_poly = self.transform(X)
        if X_poly.shape[1] > 1:
            X_poly[:, 1:] = self.normalize(X_poly[:, 1:])
        return X_poly.dot(self.W)