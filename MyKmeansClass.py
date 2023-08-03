import numpy as np


class Kmeans:
    def __init__(self, data, y, K) -> None:
        self.data = data
        self.y = y
        self.K = K
        self.initial_Centroids = np.array([self.data[np.random.randint(0, len(self.data))] for i in range(self.K)])
        _, self.data_length = data.shape
    def myEuclidean_distance(self, X, Y):
        x = np.array(X)
        y = np.array(Y)
        return np.sqrt(np.sum(np.power(x-y, 2)))

    def initial_Clusters(self):

        distances = np.zeros((len(self.data), self.K))
       
        index = 0
        for point in self.data:
            for j in range(len(self.initial_Centroids)):
                distances[index][j] = self.myEuclidean_distance(point, self.initial_Centroids[j])
            index += 1


        clusters = {}
        for i in self.initial_Centroids:
            clusters[str(i)] = []
        index2 = 0
        for point in self.data:
            clusters[str(self.initial_Centroids[distances[index2].argmin()])].append(point)
            index2 += 1
        
        return clusters


    def fit(self):
        centroids = self.initial_Centroids
        centroidsT = np.zeros((self.K,self.data_length))
        clusters = self.initial_Clusters()
        tolerance = 0.0001 # repeat until convergence.

        while not (np.linalg.norm(np.array(centroids) - np.array(centroidsT)) < tolerance):
            centroidsT = centroids.copy()
            
            #Compute the centroids of the clusters in the current partition 
            #(the centroid is the mean point of the cluster).
            i=0
            for C in clusters.keys():
                Ci = np.array(clusters[C]) 
                centroids[i] = np.mean(Ci, axis=0)
                i+=1

            
            distances = np.zeros((len(self.data), self.K))
            index = 0
            for point in self.data:
                for j in range(len(self.initial_Centroids)):
                    distances[index][j] = self.myEuclidean_distance(point, centroids[j])
                index += 1
            
            
            clusters.clear()
            for i in centroids:
                clusters[str(i)] = []

            index2 = 0
            for point in self.data:
                clusters[str(centroids[distances[index2].argmin()])].append(point)
                index2 += 1

        
        return centroidsT, clusters


