import numpy as np

class LinearRegression:
    def __init__(self, learning_rate=0.001, num_iterations=500):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.cost_history = []
        self.mean = None
        self.std = None

    def normalize(self, X, is_training=True):
        if is_training:
            self.mean = np.mean(X, axis=0)
            self.std = np.std(X, axis=0) + 1e-7
        return (X - self.mean) / self.std

    def predict(self, X):
        num_examples = X.shape[0]
        X_transform = np.append(np.ones((num_examples, 1)), X, axis=1)
        X_normalized = self.normalize(X_transform[:, 1:], is_training=False)
        X_normalized = np.insert(X_normalized, 0, 1, axis=1)
        prediction = X_normalized.dot(self.W)
        return prediction

    def update_weights(self):
        num_examples = self.X.shape[0]
        X_transform = np.append(np.ones((num_examples, 1)), self.X, axis=1)
        X_normalized = self.normalize(X_transform[:, 1:], is_training=False)
        X_normalized = np.insert(X_normalized, 0, 1, axis=1)
        Y_pred = self.predict(self.X)
        dW = - (2 * X_normalized.T.dot(self.Y - Y_pred)) / num_examples
        cost = np.sqrt(np.mean(np.square(self.Y - Y_pred)))
        self.cost_history.append(cost)
        self.W = self.W - self.learning_rate * dW

    def fit(self, X, Y):
        self.X = X
        self.Y = Y
        _, num_features = X.shape
        self.W = np.zeros((num_features + 1, 1))

        # Set normalization stats
        _ = self.normalize(X, is_training=True)

        for _ in range(self.num_iterations):
            self.update_weights()
        return self