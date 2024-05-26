import numpy as np
import pandas as pd
import sys

class gradient_descent:
    #linear regression -> the forumla is y = w0 + w1*x1 + w2*x2 + ... + wn*xn with n being the number of features

    def __init__(self, learning_rate, epochs):
        # Number of iterations
        self.epoch = epochs  # depending on epoch, parameters and training data, we will choose between stochastic, mini-batch or batch gradient descent

        # Initial guess for weights (will be set in fit method)
        self.weights = None

        # Learning rate
        self.alpha = learning_rate

    # Function to compute the Residual Sum of Squares (RSS)
    # Sigma (yj - (w0*1 + w1*x1j + w2*x2j + ... + wn*xnj))^2 for j = 1 to m with m being the training set
    def compute_rss(self, x, y):
        predictions = np.sum(self.weights * x,axis=1)
        return np.sum(np.square(y - predictions))

    # Function to compute the derivative of RSS with respect to whe single weights
    # gradient can be computed via chain rule
    # gradient of wn = -2 * xnj * (yj - (w0*1 + w1*x1j + w2*x2j + ... + wn*xnj)) for j = 1 to m with m being the training set
    def derivative_rss(self, x, y):
        y = y.reshape(-1)
        predictions = np.sum(self.weights * x,axis=1)
        errors = y - predictions
        steps = -2 * x * self.alpha
        steps = steps * errors[:, np.newaxis]
        return np.sum(steps,axis=0)

    def fit(self, x_train_input, y_train_input):
        if isinstance(x_train_input, pd.DataFrame) == True:
            x_train = x_train_input.to_numpy()
        else:
            x_train = x_train_input

        if isinstance(y_train_input, pd.DataFrame) == True:
            y_train = y_train_input.to_numpy()
        else:
            y_train = y_train_input

        self.weights = np.random.randint(1, 11, size=np.shape(x_train)[1])
        #self.weights = np.zeros(np.shape(x_train)[1])

        # Gradient Descent Algorithm
        for i in range(self.epoch):
            #batch_index = np.array(random.sample(range(1, np.shape(x_train)[0]), 200))
            # Calculate the gradient of the cost function
            #step = self.derivative_rss(x_train[batch_index], y_train[batch_index])
            step = self.derivative_rss(x_train, y_train)
            # Update weights
            self.weights = self.weights - step

        print(f"Final model with following weights for w0 ... wn:{self.weights}")

    def predict(self, x_test):
        return np.sum(self.weights * x_test,axis=1)

    def get_weights(self):
        return self.weights
