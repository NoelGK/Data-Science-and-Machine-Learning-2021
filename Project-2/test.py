from neural import Layer, Network
from aux_funcs import to_categorical, mse, mse_prime
from logistic import LogisticRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.metrics import accuracy_score

'''TESTING LOGISTIC REGRESSION'''
'''WISCONSIN BREAST CANCER DATASET'''
cancer_data = datasets.load_breast_cancer()
samples, targets = cancer_data.data, cancer_data.target.reshape(-1, 1)

X_wis_train, X_wis_test, y_wis_train, y_wis_test = train_test_split(samples, targets, test_size=0.2)

scaler = StandardScaler()
scaler.fit(X_wis_train)
X_wis_train_scaled = scaler.fit_transform(X_wis_train)
X_wis_test_scaled = scaler.fit_transform(X_wis_test)
n_wis_samples, n_wis_features = samples.shape

log_reg = LogisticRegressor(X_wis_train_scaled, y_wis_train)  # rest of the parameters are set as default
log_reg.train()  # this method performs SGD optimization with the parameters specified in the creation of the object
prediction = log_reg.predict_val(X_wis_test_scaled)  # Returns an array with the prediction for each test sample, 0 or 1

'''TESTING NEURAL NETWORK'''
'''MNIST DATASET'''
images, targets = datasets.load_digits().images, datasets.load_digits().target
n_samples = len(images)
n_cat = 10
X = images.reshape(n_samples, -1)
n_features = X.shape[1]

X_train, X_test, y_train, y_test = train_test_split(X, targets, test_size=0.2)
y_train_onehot = to_categorical(y_train, n_cat)
y_test_onehot = to_categorical(y_test, n_cat)

scaler = StandardScaler()
scaler.fit(X_train)

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

net = Network()  # Initializes network
net.use_cost(mse, mse_prime)  # Determines which cost function to use

# method self.add_layer() adds a new layer with randomly initialized params to the network.
# Make sure that each layer contains the appropiate number of inputs/outputs.
# parameter "activation" sets the activation function for each layer, default set to None. If used for regression,
# last layer must be 1 output with activation=None.
net.add_layer(Layer(n_features, 50, activation='tanh'))
net.add_layer(Layer(50, 40, activation='tanh'))
net.add_layer(Layer(40, 10, activation='tanh'))

# Method self.train(...) performs SGD optimization with the specified values of epochs, batch_size, learning rate
# and regularization. The parameter "display" is False by default and prints the error each 10 epochs.
net.train(X_train_scaled, y_train_onehot, 40, 30, 0.07, 0.01, display=True)

# For multiclass classification, use method self.predict_values(x). It returns an array with the corresponding class of
# each sample. For regression, use self.predict_probabilities(X).
# For logistic regression, either use 2 nodes in the final layer and use self.predict_values(x) or one output node with
# self.predict_probabilities(x) and then transform it to 0 and 1.
pred = net.predict_values(X_test_scaled)

print('Accuracy: ', accuracy_score(pred, y_test))
