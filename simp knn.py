import numpy as np

# Sample data
x = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

k = 3


def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


def k_nn_regression(x_train, y_train, x_query, k):
    distances = [euclidean_distance(x_query, x_train_point) for x_train_point in x_train]

    k_indices = np.argsort(distances)[:k]

    return np.mean(y_train[k_indices])



x_query = np.array([[3.5]])
prediction = k_nn_regression(x, y, x_query, k)

print(f"Prediction for {x_query.flatten()[0]}: {prediction}")
