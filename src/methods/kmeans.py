import numpy as np
from scipy.spatial.distance import cdist

"""
K = 3
Train set: accuracy = 35.283% - F1-score = 0.156169
Validation set:  accuracy = 31.073% - F1-score = 0.143040

K = 5
Train set: accuracy = 43.559% - F1-score = 0.254066
Validation set:  accuracy = 40.490% - F1-score = 0.240931

K = 10
Train set: accuracy = 67.517% - F1-score = 0.640816
Validation set:  accuracy = 68.173% - F1-score = 0.647366

K = 20
Train set: accuracy = 73.927% - F1-score = 0.705068
Validation set:  accuracy = 73.446% - F1-score = 0.697772

K = 50
Train set: accuracy = 82.638% - F1-score = 0.824791
Validation set:  accuracy = 81.544% - F1-score = 0.808888
Test set:  accuracy = 82.863% - F1-score = 0.827566

K = 100
Train set: accuracy = 83.759% - F1-score = 0.834275
Validation set:  accuracy = 80.603% - F1-score = 0.804925
Test set:  accuracy = 86.817% - F1-score = 0.868001

K = 990
Train set: accuracy = 95.146% - F1-score = 0.951438
Validation set:  accuracy = 87.006% - F1-score = 0.866922
"""


class KMeans(object):
    """
    K-Means clustering class used for cross validation. Use scipy for speedup

    We also use it to make prediction by attributing labels to clusters.
    """

    def __init__(self, K, max_iters=100):
        """
        Initialize the new object (see dummy_methods.py)
        and set its arguments.

        Arguments:
            K (int): number of clusters
            max_iters (int): maximum number of iterations
        """
        self.K = K
        self.max_iters = max_iters

    def init_centers(self, data):
        """
        Randomly pick K data points from the data as initial cluster centers.

        Arguments:
            data: array of shape (NxD) where N is the number of data points and D is the number of features (:=pixels).
            K: int, the number of clusters.
        Returns:
            centers: array of shape (KxD) of initial cluster centers
        """
        # Select the first K random index
        random_idx = np.random.permutation(data.shape[0])[:self.K]
        # Use these index to select centers from data
        centers = data[random_idx[:self.K]]

        return centers

    def compute_distance(self, data, centers):
        """
        Compute the euclidean distance between each datapoint and each center.

        Arguments:
            data: array of shape (N, D) where N is the number of data points, D is the number of features (:=pixels).
            centers: array of shape (K, D), centers of the K clusters.
        Returns:
            distances: array of shape (N, K) with the distances between the N points and the K clusters.
        """

        distances = cdist(data, centers, metric='euclidean')
        return distances

    def find_closest_cluster(self, distances):
        """
        Assign datapoints to the closest clusters.

        Arguments:
            distances: array of shape (N, K), the distance of each data point to each cluster center.
        Returns:
            cluster_assignments: array of shape (N,), cluster assignment of each datapoint, which are an integer between 0 and K-1.
        """

        cluster_assignments = np.argmin(distances, axis=1)
        return cluster_assignments

    def compute_centers(self, data, cluster_assignments):
        """
        Compute the center of each cluster based on the assigned points.

        Arguments:
            data: data array of shape (N,D), where N is the number of samples, D is number of features
            cluster_assignments: the assigned cluster of each data sample as returned by find_closest_cluster(), shape is (N,)
            K: the number of clusters
        Returns:
            centers: the new centers of each cluster, shape is (K,D) where K is the number of clusters, D the number of features
        """
        centers = np.array([np.mean(data[cluster_assignments == k], axis=0) for k in range(self.K)])

        return centers

    def k_means(self, data, max_iter):
        """
        Main K-Means algorithm that performs clustering of the data.
        
        Arguments: 
            data (array): shape (N,D) where N is the number of data samples, D is number of features.
            max_iter (int): the maximum number of iterations
        Returns:
            centers (array): shape (K,D), the final cluster centers.
            cluster_assignments (array): shape (N,) final cluster assignment for each data point.
        """

        # Initialize the centers
        centers = self.init_centers(data)

        # Loop over the iterations
        for i in range(max_iter):
            if (i + 1) % 10 == 0:
                print(f"Iteration {i + 1}/{max_iter}...")
            old_centers = centers.copy()  # keep in memory the centers of the previous iteration

            distances = self.compute_distance(data, old_centers)
            cluster_assignments = self.find_closest_cluster(distances)
            centers = self.compute_centers(data, cluster_assignments)

            # End of the algorithm if the centers have not moved
            if np.all(old_centers == centers):
                print(f"K-Means has converged after {i + 1} iterations!")
                break

        # Compute the final cluster assignments
        distances = self.compute_distance(data, centers)
        cluster_assignments = self.find_closest_cluster(distances)

        return centers, cluster_assignments

    def assign_labels_to_centers(self, cluster_assignments, true_labels):
        """
        Use voting to attribute a label to each cluster center.

        Arguments:
            centers: array of shape (K, D), cluster centers
            cluster_assignments: array of shape (N,), cluster assignment for each data point.
            true_labels: array of shape (N,), true labels of data
        Returns:
            cluster_center_label: array of shape (K,), the labels of the cluster centers
        """

        cluster_center_label = np.array([np.argmax(np.bincount(true_labels[cluster_assignments == k]))
                                         for k in range(self.K)])
        return cluster_center_label

    def fit(self, training_data, training_labels):
        """
        Train the model and return predicted labels for training data.

        You will need to first find the clusters by applying K-means to
        the data, then to attribute a label to each cluster based on the labels.
        
        Arguments:
            training_data (array): training data of shape (N,D)
            training_labels (array): labels of shape (N,)
        Returns:
            pred_labels (array): labels of shape (N,)
        """
        self.final_centers, cluster_assignments = self.k_means(training_data, self.max_iters)
        self.cluster_center_label = self.assign_labels_to_centers(cluster_assignments, training_labels)

        return self.predict(training_data)

    def predict(self, test_data):
        """
        Runs prediction on the test data given the cluster center and their labels.

        To do this, first assign data points to their closest cluster, then use the label
        of that cluster as prediction.
        
        Arguments:
            test_data (np.array): test data of shape (N,D)
        Returns:
            pred_labels (np.array): labels of shape (N,)
        """
        # Compute cluster assignments
        distances = self.compute_distance(test_data, self.final_centers)
        cluster_assignments = self.find_closest_cluster(distances)

        # Convert cluster index to label
        pred_labels = self.cluster_center_label[cluster_assignments]

        return pred_labels
