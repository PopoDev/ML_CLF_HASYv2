import numpy as np

## MS2

"""
python main.py --data dataset  --method nn --nn_type mlp --lr 1e-2 --max_iters 100 --use_pca --pca_d 200
pca_d=200
[PCA] Train set: accuracy = 99.735% - F1-score = 0.997112
[PCA] Validation set:  accuracy = 84.429% - F1-score = 0.836122

pca_d=256
Train set: accuracy = 99.886% - F1-score = 0.998817
Validation set:  accuracy = 86.390% - F1-score = 0.858670

pca_d=400
[PCA] Train set: accuracy = 100.000% - F1-score = 1.000000
[PCA] Validation set:  accuracy = 87.428% - F1-score = 0.872546

pca_d=512
[PCA] Train set: accuracy = 100.000% - F1-score = 1.000000
[PCA] Validation set:  accuracy = 88.120% - F1-score = 0.875139
[PCA] Test set:  accuracy = 88.812% - F1-score = 0.881041

Train set: accuracy = 100.000% - F1-score = 1.000000
Validation set:  accuracy = 87.659% - F1-score = 0.872662
Test set:  accuracy = 89.965% - F1-score = 0.892299

ADAM
python main.py --data dataset  --method nn --nn_type mlp --lr 1e-3 --max_iters 100 --test
Train set: accuracy = 100.000% - F1-score = 1.000000
Test set:  accuracy = 91.696% - F1-score = 0.911778

python main.py --data dataset  --method nn --nn_type mlp --lr 1e-3 --max_iters 100 --use_pca --pca_d 100 --test
[PCA] Train set: accuracy = 100.000% - F1-score = 1.000000
[PCA] Test set:  accuracy = 92.272% - F1-score = 0.918111

python main.py --data dataset  --method kmeans --K 400 --test
Train set: accuracy = 79.772% - F1-score = 0.791002
Test set:  accuracy = 79.585% - F1-score = 0.788929

python main.py --data dataset  --method kmeans --K 400 --use_pca --pca_d 100 --test
[PCA] Train set: accuracy = 84.907% - F1-score = 0.842663
[PCA] Test set:  accuracy = 82.238% - F1-score = 0.815201

python main.py --data dataset  --method logistic_regression --lr 1e-3 --max_iters 200 --test
Train set: accuracy = 100.000% - F1-score = 1.000000
Test set:  accuracy = 85.121% - F1-score = 0.838993

python main.py --data dataset  --method logistic_regression --lr 1e-3 --max_iters 200 --use_pca --pca_d 100 --test
[PCA] Train set: accuracy = 100.000% - F1-score = 1.000000
[PCA] Validation set:  accuracy = 83.737% - F1-score = 0.834170
[PCA] Test set:  accuracy = 86.621% - F1-score = 0.857901

python main.py --data dataset --method svm --svm_c 1000 --svm_kernel rbf --svm_gamma 0.0005 --test
Train set: accuracy = 100.000% - F1-score = 1.000000
Test set:  accuracy = 92.388% - F1-score = 0.919201

python main.py --data dataset --method svm --svm_c 1000 --svm_kernel rbf --svm_gamma 0.0005 --use_pca --pca_d 100 --test
[PCA] Train set: accuracy = 100.000% - F1-score = 1.000000
[PCA] Test set:  accuracy = 93.195% - F1-score = 0.930295
"""


class PCA(object):
    """
    PCA dimensionality reduction class.
    
    Feel free to add more functions to this class if you need,
    but make sure that __init__(), find_principal_components(), and reduce_dimension() work correctly.
    """

    def __init__(self, d):
        """
        Initialize the new object (see dummy_methods.py)
        and set its arguments.

        Arguments:
            d (int): dimensionality of the reduced space
        """
        self.d = d
        
        # the mean of the training data (will be computed from the training data and saved to this variable)
        self.mean = None 
        # the principal components (will be computed from the training data and saved to this variable)
        self.W = None

    def find_principal_components(self, training_data):
        """
        Finds the principal components of the training data and returns the explained variance in percentage.

        IMPORTANT: 
            This function should save the mean of the training data and the kept principal components as
            self.mean and self.W, respectively.

        Arguments:
            training_data (array): training data of shape (N,D)
        Returns:
            exvar (float): explained variance of the kept dimensions (in percentage, i.e., in [0,100])
        """

        # Compute the mean of data
        self.mean = np.mean(training_data, axis=0)
        # print("Mean:", mean)

        # Create the covariance matrix
        cov = np.cov(training_data.T)
        # print("cov:", cov.shape, cov)

        # Compute the eigenvectors and eigenvalues. Hint: look into np.linalg.eigh()
        eigvals, eigvecs = np.linalg.eigh(cov)
        # print("eigvals", eigvals.shape, eigvals)
        # print("eigvecs", eigvecs.shape, eigvecs)

        # Choose the top d eigenvalues and corresponding eigenvectors.
        # Hint: sort the eigenvalues (with corresponding eigenvectors) in decreasing order first.
        decreasing = eigvals.argsort()[::-1]
        decreasing_eigvals = eigvals[decreasing]
        decreasing_eigvecs = eigvecs[:, decreasing]  # Select column in decreasing order
        # print("decreasing_eigvals", decreasing_eigvals)
        # print("decreasing_eigvecs", decreasing_eigvecs)

        d_eigvals = decreasing_eigvals[:self.d]
        d_eigvecs = decreasing_eigvecs[:, :self.d]  # Select column
        # print("d_eigvecs", d_eigvecs.shape, d_eigvecs)

        self.W = d_eigvecs
        # print("self.W", self.W.shape, self.W)
        eg = d_eigvals

        # Compute the explained variance
        exvar = np.sum(eg) * 100 / np.sum(eigvals)

        return exvar

    def reduce_dimension(self, data):
        """
        Reduce the dimensionality of the data using the previously computed components.

        Arguments:
            data (array): data of shape (N,D)
        Returns:
            data_reduced (array): reduced data of shape (N,d)
        """

        # Center the data with the mean
        centered = data - self.mean
        # print("centered", centered.shape, centered)

        data_reduced = centered @ self.W
        # print("self.W", self.W.shape, self.W)

        return data_reduced
        

