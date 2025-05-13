import pandas as pd
import numpy as np

class my_KMeans:
    def __init__(self, n_clusters=5, n_init=10, max_iter=300, tol=1e-4):
        self.n_clusters = int(n_clusters)
        self.n_init = n_init
        self.max_iter = int(max_iter)
        self.tol = tol
        self.cluster_centers_ = None
        self.sse_ = None

    @staticmethod
    def euclidean_distance(data_point, centroids):
        return np.sqrt(np.sum((centroids - data_point) ** 2, axis=1))

    def fit(self, X):
        best_sse = float('inf')
        best_centroids = None

        X = X.to_numpy()  

        for _ in range(self.n_init):
            centroids = np.random.uniform(
                np.amin(X, axis=0),
                np.amax(X, axis=0),
                size=(self.n_clusters, X.shape[1])
            )

            for _ in range(self.max_iter):
                y = []
                
                for data_point in X:
                    distances = self.euclidean_distance(data_point, centroids)
                    cluster_num = np.argmin(distances)
                    y.append(cluster_num)

                y = np.array(y)


                new_centroids = []
                for i in range(self.n_clusters):
                    cluster_points = X[y == i]
                    if len(cluster_points) == 0:
                        new_centroids.append(centroids[i])
                    else:
                        new_centroids.append(np.mean(cluster_points, axis=0))

                new_centroids = np.array(new_centroids)


                if np.max(np.abs(centroids - new_centroids)) < self.tol:
                    break

                centroids = new_centroids

            
            sse = np.sum([np.min(self.euclidean_distance(data_point, centroids)) ** 2 for data_point in X])

            
            if sse < best_sse:
                best_sse = sse
                best_centroids = centroids

        self.cluster_centers_ = best_centroids
        self.sse_ = best_sse

    def predict(self, X):
        X = X.to_numpy()  
        predictions = [np.argmin(self.euclidean_distance(data_point, self.cluster_centers_)) for data_point in X]
        return np.array(predictions)

    def transform(self, X):
        X = X.to_numpy()  
        distances = [self.euclidean_distance(data_point, self.cluster_centers_) for data_point in X]
        return np.array(distances)

    def fit_predict(self, X):
        self.fit(X)
        return self.predict(X)
