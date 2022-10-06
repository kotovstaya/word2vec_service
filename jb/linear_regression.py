import typing as tp
import numpy as np
from copy import deepcopy


class LinearRegression:
    def __init__(self,
                 lr: float,
                 iterations: tp.Optional[int] = None,
                 epsilon: tp.Optional[float] = None):
        self.w = None
        self.b = None
        self.lr = lr
        self.iterations = iterations
        self.epsilon = epsilon

        self.rows = None
        self.cols = None

    def _init_weights(self, cols: int) -> None:
        self.w = np.zeros(shape=(cols,))
        self.b = 0

    def fit(self, X: np.array, y: np.array) -> None:
        self.rows, self.cols = X.shape
        self._train(X, y)

    def _train(self, X: np.array, y: np.array):
        self._init_weights(self.cols)
        print(self.w)
        self._update_weights(X, y)
        print(self.w)

    def predict(self, X: np.array):
        return np.dot(X, self.w) + self.b

    def _update_in_iterations(self, X: np.array, y_true: np.array):
        for _ in range(self.iterations):
            self._update(X, y_true)

    def _update(self, X: np.array, y_true: np.array):
        y_pred = self.predict(X)
        new_w = - 2 * X.T.dot(y_true - y_pred) / self.rows
        self.w -= self.lr * new_w

        new_b = - 2 * np.sum(y_true - y_pred) / self.rows
        self.b -= self.lr * new_b

    def _get_epsilon(self):
        return np.linalg.norm(self.w)

    def _update_by_epsilon(self, X: np.array, y_true: np.array):
        prev_w = deepcopy(self.w)
        cur_w = deepcopy(self.w + 100)
        while np.linalg.norm(cur_w - prev_w) > self.epsilon:
            prev_w = deepcopy(cur_w)
            self._update(X, y_true)
            cur_w = deepcopy(self.w)

    def _update_weights(self, X: np.array, y_true: np.array):
        if self.iterations is not None:
            self._update_in_iterations(X, y_true)
        elif self.epsilon is not None:
            self._update_by_epsilon(X, y_true)
        else:
            raise Exception("You must choose one method")


if __name__ == "__main__":
    X = np.random.normal(0, 1, size=(100, 4))
    y = 0.5 * X[:, 0] + 0.3 * X[:, 1] - 0.2 * X[:, 2] + 0.8 * X[:, 3]
    # linreg = LinearRegression(lr=0.1, iterations=100)
    linreg = LinearRegression(lr=0.1, epsilon=1e-8)
    linreg.fit(X,y)
