import numpy as np
from models.linear_regression_model import LinearRegression

class RidgeRegression(LinearRegression):
    def __init__(self, learning_rate, num_iterations, l2_penalty):
        self.l2_penalty = l2_penalty
        super().__init__(learning_rate, num_iterations)

    def update_weights(self):
        num_examples, _ = self.X.shape

        X_transform = np.append(np.ones((num_examples, 1)), self.X, axis=1)
        X_normalized = self.normalize(X_transform[:, 1:])
        X_normalized = np.insert(X_normalized, 0, 1, axis=1)

        Y_pred = self.predict(self.X)

        # Ridge penalty added here
        dW = - (2 * (X_normalized.T.dot(self.Y - Y_pred)) + 2 * self.l2_penalty * self.W) / num_examples
        cost = np.sqrt(np.mean((self.Y - Y_pred) ** 2))
        self.cost_history.append(cost)

        self.W = self.W - self.learning_rate * dW

        return self