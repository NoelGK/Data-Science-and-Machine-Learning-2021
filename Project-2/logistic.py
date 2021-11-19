import numpy as np
from aux_funcs import sigmoid


class LogisticRegressor:
    def __init__(self, X_data, y_data, activation_func=sigmoid, n_eps=100, batch_size=50, eta=0.01, lmd=0.1):
        # Data used for SGD training
        self.X_data = None
        self.y_data = None

        # Full data
        self.X_data_full = X_data
        self.y_data_full = y_data

        self.n_samples = self.X_data_full.shape[0]
        self.n_features = self.X_data_full.shape[1]

        self.activation_func = activation_func

        # SGD parameters
        self.n_eps = n_eps
        self.batch_size = batch_size
        self.n_iter = int(self.n_samples / self.batch_size)
        self.eta = eta
        self.lmd = lmd

        # Initialize parameters
        self.beta = np.random.randn(self.n_features, 1)
        self.gradient = None

        # Outputs
        self.probs = None

    def predict_val(self, X):
        n_samples = X.shape[0]
        mat = np.matmul(X, self.beta)
        probs = self.activation_func(mat)
        output = np.zeros(n_samples).reshape(-1, 1)

        for i in range(n_samples):
            if probs[i] > 0.5:
                output[i] = 1
            else:
                output[i] = 0

        return output

    def get_grad(self):
        mat = np.matmul(self.X_data, self.beta)
        self.probs = self.activation_func(mat)
        self.gradient = np.matmul(self.X_data.T, self.probs - self.y_data)

        if self.lmd > 0.0:
            self.gradient -= self.lmd * self.beta

    def train(self):
        indx = np.arange(self.n_samples)

        for k in range(self.n_eps):
            for j in range(self.n_iter):
                chosen_data = np.random.choice(indx, size=self.batch_size, replace=False)
                self.X_data = self.X_data_full[chosen_data]
                self.y_data = self.y_data_full[chosen_data]
                self.get_grad()

                self.beta -= self.eta * self.gradient
