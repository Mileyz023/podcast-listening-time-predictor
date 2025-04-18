from sklearn.linear_model import LinearRegression as SklearnLinearRegression

# 1. Train sklearn Linear Regression
sk_model = SklearnLinearRegression()
sk_model.fit(X_train, y_train)

# 2. Wrap the model to make it compatible with evaluate_model
class SKModelWrapper:
    def __init__(self, model):
        self.model = model
        self.cost_history = []  # not used, but needed for compatibility

    def predict(self, X):
        return self.model.predict(X)

# 3. Evaluate with the same function
evaluate_model(SKModelWrapper(sk_model), X_train, y_train, X_eval, y_eval)