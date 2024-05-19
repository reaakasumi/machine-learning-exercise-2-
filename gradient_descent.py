import numpy as np

# Sample data
# x represents some feature, y is the target
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 6, 8, 10])

# Initial guess for w1
w1 = 10

# Learning rate
alpha = 0.01


# Function to compute the Residual Sum of Squares (RSS)
def compute_rss(w1, x, y):
    predictions = w1 * x
    return np.sum((y - predictions) ** 2)


# Function to compute the derivative of RSS with respect to w1
def derivative_rss_w1(w1, x, y):
    predictions = w1 * x
    errors = y - predictions
    return -2 * np.sum(errors * x)


# Number of iterations
iterations = 100

# Gradient Descent Algorithm
for i in range(iterations):
    # Calculate the gradient of the cost function
    grad = derivative_rss_w1(w1, x, y)

    # Update w1
    w1 = w1 - alpha * grad

    # Optionally print the cost every 10 iterations
    if i % 10 == 0:
        rss = compute_rss(w1, x, y)
        print(f"Iteration {i}: w1 = {w1}, RSS = {rss}")

print(f"Final model: y = {w1} * x")
