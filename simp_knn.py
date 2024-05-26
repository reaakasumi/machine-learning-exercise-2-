import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics.pairwise import manhattan_distances

class custom_knn:
    def __init__(self):
        self.k = None
        self.x_train = None
        self.y_train = None

    #distance metrics:
    #1 = euclician
    #2 = cosine_distance
    #3 = manhattan_distance
    def fit(self, x_train, y_train, k = 5):
        self.k = k
        self.x_train = x_train
        self.y_train = y_train.to_numpy()

    def predict(self, x_test, distance_metric = 1):
        if(distance_metric == 1):
            distances = euclidean_distances(x_test, self.x_train)
        elif(distance_metric == 2):
            distances = cosine_distances(x_test, self.x_train)
        elif(distance_metric == 3):
            distances = manhattan_distances(x_test, self.x_train)
        else:
            print("Invalid distance metric")
            return
        k_indices = np.argsort(distances, axis=1)[:, :self.k]

        return np.mean(self.y_train[k_indices], axis=1)

