import pandas as pd
import numpy as np
from collections import Counter

# Changed metric to integers 1:3 to avoid misspelling
class my_KNN:

    def __init__(self, n_neighbors=5, metric=1, p=2):
        # Metric = {1: 'minkowski', 2: 'euclidean', 3: 'manhattan'}
        self.n_neighbors = int(n_neighbors)
        self.metric = metric
        self.p = p
        if self.metric not in [1, 2, 3]:
            raise ValueError("Invalid metric: Use 1 for 'minkowski', 2 for 'euclidean', or 3 for 'manhattan'")

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        self.classes_ = list(set(y))

    def dist_equation(self, train_value, test_value):
        if self.metric == 1:
            # minkowski
            distance = np.sum(np.abs(train_value - test_value)**self.p)**(1/self.p)
        elif self.metric == 2:
            #euclidean
            distance = np.sqrt(np.sum((train_value - test_value)**2))
        elif self.metric == 3:
            # manhatta
            distance = np.sum(np.abs(train_value - test_value))
        
        return distance

    def get_neighbors(self, test_value):
        distances = []
        for index, train_value in self.X_train.iterrows():
            dist = self.dist_equation(train_value, test_value)
            distances.append((index, dist))
        distances.sort(key=lambda x: x[1])
        neighbors = [self.y_train[index] for index, dist in distances[:self.n_neighbors]]
        
        return neighbors

    def predict(self, X):
        predictions = []
        for index, test_value in X.iterrows():
            neighbors = self.get_neighbors(test_value)
            post_probs = {label: 0 for label in self.classes_}
            for neighbor in neighbors:
                post_probs[neighbor] += 1
            prediction = max(post_probs, key=post_probs.get)
            predictions.append(prediction)
        
        return predictions

    def predict_proba(self, X):
        prob_list = []
        for index, test_value in X.iterrows():
            neighbors = self.get_neighbors(test_value)
            post_probs = {label: 0 for label in self.classes_} 
            for neighbor in neighbors:
                post_probs[neighbor] += 1
            total_neighbors = sum(post_probs.values())
            normalized_probs = {c: count / total_neighbors for c, count in post_probs.items()}
            prob_list.append(normalized_probs)

        return pd.DataFrame(prob_list, columns=self.classes_)
