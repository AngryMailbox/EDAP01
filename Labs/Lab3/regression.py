import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import sklearn as sk


def __init__(self):
    pass


def normal(arr):
    """
    Normalizes an array to the range [0, 1]
    :param arr: The array to normalize
    :return: The normalized array, the minimum value and the maximum value
    """
    min_val = np.min(arr)
    max_val = np.max(arr)
    return ((arr - min_val) / (max_val - min_val), min_val, max_val)


def sse(X, y, w):
    """
    Calculates the sum of squared errors
    :param X: The input matrix
    :param y: The output vector
    :param w: The weights
    :return: The sum of squared errors
    """
    error = y - X @ w
    return error.T @ error


def predict(X, w):
    """
    Predicts the output of a linear model by multiplying the input matrix with the weights
    :param X: The input matrix
    :param w: The weights
    :return: The predicted output
    """
    return X @ w


def fit_stoch(X, y, alpha, w, epochs=500, epsilon=1.0e-5):
    """
    Stochastic gradient descent. This function is used to train a linear model using the stochastic gradient descent.
    :param X: The input matrix
    :param y: The output vector
    :param alpha: The learning rate
    :param w: The initial weights
    :param epochs: The number of epochs (an epoch is a complete pass through the dataset)
    :param epsilon: The threshold for the norm of the gradient
    :return: The trained weights
    """
    global logs, logs_stoch
    logs = []
    logs_stoch = []
    np.random.seed(0)
    idx = list(range(len(X)))
    for epoch in range(epochs):
        np.random.shuffle(idx)
        for i in idx:
            # error is a scalar
            error = (predict(X[i], w) - y[i])[0]
            gradient = error * X[i : i + 1].T
            w = w - alpha * gradient
            logs_stoch += (w, alpha, sse(X, y, w))
        if np.linalg.norm(gradient) < epsilon:
            break
        logs += (w, alpha, sse(X, y, w))
    print("Epoch", epoch)
    return w


def fit_batch(X, y, alpha, w, epochs=500, epsilon=1.0e-5):
    """
    Batch gradient descent. This function is used to train a linear model using the batch gradient descent.
    :param X: The input matrix
    :param y: The output vector
    :param alpha: The learning rate
    :param w: The initial weights
    :param epochs: The number of epochs
    :param epsilon: The threshold for the norm of the gradient
    :return: The trained weights
    """
    global logs
    logs = []
    alpha /= len(X)
    for epoch in range(epochs):
        error = predict(X, w) - y
        gradient = X.T @ error
        w = w - alpha * gradient
        logs += (w, alpha, sse(X, y, w))
        if np.linalg.norm(gradient) < epsilon:
            break
    print("Epoch", epoch)
    return w


def normalize(Xy):
    max = np.amax(Xy, axis=0)
    Xy = 1 / max * Xy
    return (Xy, max)


def normalize2(X):
    return (X - np.mean(X, axis=0)) / np.std(X, axis=0), np.max(X, axis=0)


def apply_normalize(X, y, normalized=True, debug=False):
    """
    Normalizes the input and output data
    :param X: The input matrix
    :param y: The output vector
    :param normalized: A flag to indicate whether the data should be normalized
    :param debug: A flag to indicate whether to print debug information
    :return: The normalized input matrix, the normalized output vector, the learning rate and the maximum values
    """
    # Predictors
    X = np.array(X)
    # Response
    y = np.array([y]).T

    alpha = 1.0e-10
    if normalized:
        X, maxima_X = normalize(X)
        y, maxima_y = normalize(y)
        maxima = np.concatenate((maxima_X, maxima_y))
        alpha = 1.0
        print("-Normalized-")
    return X, y, alpha, maxima


def apply_batch(X_train, y_train, alpha, normalized=False, debug=False):
    """
    Applies the batch gradient descent to the training data
    :param X_train: The input matrix
    :param y_train: The output vector
    :param alpha: The learning rate
    :param normalized: A flag to indicate whether the data should be normalized
    :param debug: A flag to indicate whether to print debug information
    :return: TODO: Add return
    """
    print("===Batch descent===")
    w = np.zeros((X_train.shape[1], 1))
    w = fit_batch(X_train, y_train, alpha, w)
    print("Weights", w)
    print("SSE", sse(X_train, y_train, w))
    if normalized:
        maxima = maxima.reshape(-1, 1)
        w = maxima[-1, 0] * (w / maxima[:-1, 0:1])
        print("Restored weights", w)
    if debug:
        print("Logs", logs)


# apply_batch(X_train, y_train, alpha, normalized=False, debug=False)
