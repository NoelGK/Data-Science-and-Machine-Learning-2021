import numpy as np
from aux_funcs import sigmoid, sigmoid_prime, tanh, tanh_prime
from aux_funcs import relu, relu_prime, leakyrelu, leaky_prime


class Layer:
    def __init__(self, n_inputs, n_outputs, activation=None):
        self.weigths = np.random.rand(n_inputs, n_outputs) - 0.5
        self.bias = np.random.rand(n_outputs) - 0.5

        acts = ['sigmoid', 'tanh', 'relu', 'leaky_relu']

        if activation is None:
            self.activation = lambda t: t
            self.activation_prime = lambda t: 1

        elif activation == 'sigmoid':
            self.activation = sigmoid
            self.activation_prime = sigmoid_prime

        elif activation == 'tanh':
            self.activation = tanh
            self.activation_prime = tanh_prime

        elif activation == 'relu':
            self.activation = relu
            self.activation_prime = relu_prime

        elif activation == 'leaky_relu':
            self.activation = leakyrelu
            self.activation_prime = leaky_prime

        else:
            raise Exception(activation+f' activation function not implemented. Try one of the following: {acts}')

        self.input = None
        self.output = None

    def feed_forward(self, input_data):
        self.input = input_data
        matrix = np.matmul(self.input, self.weigths) + self.bias
        self.output = self.activation(matrix)

        return self.output

    def backpropagate(self, output_error, eta, lmd):
        delta = np.matmul(output_error, self.weigths.T) * self.activation_prime(self.input)
        weigths_grad = np.matmul(self.input.T, output_error)
        bias_grad = np.sum(output_error, axis=0)

        if lmd > 0.0:
            weigths_grad += lmd * self.weigths
            bias_grad += lmd * self.bias

        self.weigths -= eta * weigths_grad
        self.bias -= eta * bias_grad

        return delta


class Network:
    def __init__(self):
        self.layers = []
        self.cost = None
        self.cost_prime = None
        self.n_categories = None

    def use_cost(self, cost, cost_prime):
        self.cost = cost
        self.cost_prime = cost_prime

    def add_layer(self, layer):
        self.layers.append(layer)

    def train(self, x_train, y_train, epochs, batch, eta, lmd=0.0, display=False):
        n_samples = x_train.shape[0]
        indx = np.arange(n_samples)
        self.n_categories = y_train.shape[1]

        for i in range(epochs):
            err = 0.0
            for j in range(n_samples):
                chosen_data = np.random.choice(indx, size=batch, replace=False)
                output = x_train[chosen_data]
                for layer in self.layers:
                    output = layer.feed_forward(output)

                err += np.sum(self.cost(y_train[chosen_data], output))

                error = self.cost_prime(y_train[chosen_data], output)
                for layer in reversed(self.layers):
                    error = layer.backpropagate(error, eta, lmd)

            err /= n_samples

            if display and i % 10 == 0.0:
                print(f'Epoch {i+1}, error={err}')

    def predict_values(self, X):
        n = len(X)
        result = np.zeros((n, 1))
        for i in range(n):
            output = X[i]
            for layer in self.layers:
                output = layer.feed_forward(output)

            result[i] = np.argmax(output)

        return result

    def predict_probabilities(self, X):
        n = len(X)
        result = np.zeros((n, self.n_categories))
        for i in range(n):
            output = X[i]
            for layer in self.layers:
                output = layer.feed_forward(output)
            result[i] = output

        return result
