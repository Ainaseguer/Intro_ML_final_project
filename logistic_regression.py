# This code is copied from https://www.geeksforgeeks.org/machine-learning/ml-kaggle-breast-cancer-wisconsin-diagnosis-using-logistic-regression/
# Only the preprocessing steps have been adjusted to fit our format of the dataset.

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

# Loading and preprocessing the data
data = pd.read_csv("data_given.data", header=None)

y = data[1].map({"M": 1, "B": 0}).values
x_data = data.drop([0, 1], axis=1)

#split data before normalisation to avoid leakage 
x_train, x_test, y_train, y_test = train_test_split(
    x_data, y, test_size=0.15, random_state=42
)

#Min-max scaling on train only
train_min = x_train.min()
train_max = x_train.max()


x_train = (x_train - train_min) / (train_max - train_min)
x_test = (x_test - train_min) / (train_max - train_min)

#PCA in the train data and transpose test 
pca = PCA(n_components=0.95, random_state=42)
x_train = pca.fit_transform(x_train.to_numpy())
x_test = pca.transform(x_test.to_numpy())

x_train = x_train.T
x_test = x_test.T
y_train = y_train.T
y_test = y_test.T


# Initializing parameters
def initialize_weights_and_bias(dimension):
    w = np.random.randn(dimension, 1) * 0.01
    b = 0.0

    return w, b


# Calculating the sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# Forward and Backward propagation
def forward_backward_propagation(w, b, x_train, y_train):
    m = x_train.shape[1]
    z = np.dot(w.T, x_train) + b
    y_head = sigmoid(z)

    cost = (-1 / m) * np.sum(
        y_train * np.log(y_head) + (1 - y_train) * np.log(1 - y_head)
    )

    derivative_weight = (1 / m) * np.dot(x_train, (y_head - y_train).T)
    derivative_bias = (1 / m) * np.sum(y_head - y_train)

    gradients = {
        "derivative_weight": derivative_weight,
        "derivative_bias": derivative_bias,
    }

    return cost, gradients


# Updating weights and bias
def update(w, b, x_train, y_train, learning_rate, num_iterations):
    costs = []
    gradients = {}
    for i in range(num_iterations):
        cost, grad = forward_backward_propagation(w, b, x_train, y_train)
        w -= learning_rate * grad["derivative_weight"]
        b -= learning_rate * grad["derivative_bias"]

        if i % 100 == 0:
            costs.append(cost)
            print(f"Cost after iteration {i}: {cost}")

    parameters = {"weight": w, "bias": b}
    return parameters, gradients, costs


# Making predictions
def predict(w, b, x_test):
    m = x_test.shape[1]
    y_prediction = np.zeros((1, m))
    z = sigmoid(np.dot(w.T, x_test) + b)

    for i in range(z.shape[1]):
        y_prediction[0, i] = 1 if z[0, i] > 0.5 else 0

    return y_prediction


# Logistic Regression Model
def logistic_regression(
    x_train, y_train, x_test, y_test, learning_rate=0.01, num_iterations=1000
):
    dimension = x_train.shape[0]
    w, b = initialize_weights_and_bias(dimension)
    parameters, gradients, costs = update(
        w, b, x_train, y_train, learning_rate, num_iterations
    )

    y_prediction_test = predict(parameters["weight"], parameters["bias"], x_test)
    y_prediction_train = predict(parameters["weight"], parameters["bias"], x_train)

    print(
        f"Train accuracy: {100 - np.mean(np.abs(y_prediction_train - y_train)) * 100}%"
    )
    print(f"Test accuracy: {100 - np.mean(np.abs(y_prediction_test - y_test)) * 100}%")


logistic_regression(
    x_train, y_train, x_test, y_test, learning_rate=0.01, num_iterations=1000
)
