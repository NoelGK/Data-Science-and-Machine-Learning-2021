import numpy as np


"""ACTIVATION FUNCTIONS: sigmoid, tanh, ReLU, Leaky ReLU"""


def sigmoid(z):
    return 1/(1 + np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))


def tanh(x):
    return np.tanh(x)


def tanh_prime(x):
    return 1 - np.tanh(x)**2


def relu(x):
    return np.maximum(x, 0.0)


def relu_prime(x):
    r = np.zeros(x.shape)
    r[x >= 0.0] = 1.0
    r[x < 0.0] = 0.0
    return r


def leakyrelu(x):
    r = np.zeros(x.shape)
    r[x >= 0.0] = x[x >= 0.0]
    r[x < 0.0] = 0.01 * x[x < 0.0]
    return r


def leaky_prime(x):
    r = np.zeros(x.shape)
    r[x >= 0.0] = 1.0
    r[x < 0.0] = -0.01
    return r


def softmax(z):
    exp_fact = np.exp(z)
    s = np.sum(exp_fact, axis=1, keepdims=True)
    return exp_fact/s


def stablesoftmax(z):
    shift = z - np.max(z)
    exps = np.exp(shift)
    return exps/np.sum(exps)


def to_categorical(int_vector, n_categories):
    n_inputs = len(int_vector)
    onehot_vector = np.zeros((n_inputs, n_categories))
    onehot_vector[range(n_inputs), int_vector] = 1.0
    return onehot_vector


def FrankeFunc(x, y):
    s1 = 0.75 * np.exp(-0.25 * (9 * x - 2) ** 2 - 0.25 * (9 * y - 2) ** 2)
    s2 = 0.75 * np.exp(-(9 * x - 2) ** 2 / 49 - (9 * y - 2) ** 2 / 10)
    s3 = 0.5 * np.exp(-0.25 * (9 * x - 7) ** 2 - 0.25 * (9 * y - 3) ** 2)
    s4 = -0.2 * np.exp(-(9 * x - 4) ** 2 - (9 * y - 7) ** 2)

    return s1 + s2 + s3 + s4


# Mean squared error function
def mse(y_true, y_pred):
    return np.mean(np.power(y_true-y_pred, 2))


def mse_prime(y_true, y_pred):
    return 2*(y_pred-y_true)/y_true.shape[0]
