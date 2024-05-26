import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics.pairwise import manhattan_distances

class custom_knn:
    def __init__(self, k = 5, distance_metric = 1):
        self.distance_metric = distance_metric
        self.k = k
        self.x_train = None
        self.y_train = None

    #distance metrics:
    #1 = euclician
    #2 = cosine_distance
    #3 = manhattan_distance
    def fit(self, x_train_input, y_train_input):
        if isinstance(x_train_input, pd.DataFrame) == True:
            self.x_train = x_train_input.to_numpy()
        else:
            self.x_train = x_train_input

        if isinstance(y_train_input, pd.DataFrame) == True:
            self.y_train = y_train_input.to_numpy()
        else:
            self.y_train = y_train_input

    def predict(self, x_test):
        result = np.array([])
        print(x_test.shape)
        print(x_test[0])

        for i in range(x_test.shape[0]):
            np.append(result, self._predict_single([x_test[i]], self.distance_metric))

        return result

    def _predict_single(self, x_test, distance_metric):
        #print("ADASDAS")
        #print(x_test)
        if distance_metric == 1:
            distances = euclidean_distances(x_test, self.x_train)
        elif distance_metric == 2:
            distances = cosine_distances(x_test, self.x_train)
        elif distance_metric == 3:
            distances = manhattan_distances(x_test, self.x_train)
        else:
            print("Invalid distance metric")
            return
        k_indices = np.argsort(distances, axis=1)[:, :self.k]

        return np.mean(self.y_train[k_indices], axis=1)

