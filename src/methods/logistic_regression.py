import numpy as np

from ..utils import get_n_classes, label_to_onehot, onehot_to_label, accuracy_fn

"""
python main.py --data dataset  --method logistic_regression --lr 1e-5 --max_iters 100
Train set: accuracy = 65.339% - F1-score = 0.642665
Validation set:  accuracy = 58.945% - F1-score = 0.577517

python main.py --data dataset  --method logistic_regression --lr 1e-4 --max_iters 100
Train set: accuracy = 90.230% - F1-score = 0.901054
Validation set:  accuracy = 84.746% - F1-score = 0.843599
Test set:  accuracy = 89.077% - F1-score = 0.891435

python main.py --data dataset  --method logistic_regression --lr 1e-3 --max_iters 200

Train set: accuracy = 100.000% - F1-score = 1.000000
Validatoin set:  accuracy = 89.831% - F1-score = 0.896810
Test set:  accuracy = 92.467% - F1-score = 0.921660
"""


class LogisticRegression(object):
    """
    Logistic regression classifier.
    """

    def __init__(self, lr, max_iters=500):
        """
        Initialize the new object (see dummy_methods.py)
        and set its arguments.

        Arguments:
            lr (float): learning rate of the gradient descent
            max_iters (int): maximum number of iterations
        """
        self.lr = lr
        self.max_iters = max_iters

    def f_softmax(self, data, w):
        """
        Softmax function for multi-class logistic regression.

        Args:
            data (array): Input data of shape (N, D)
            w (array): Weights of shape (D, C) where C is the number of classes
        Returns:
            array of shape (N, C): Probability array where each value is in the
                range [0, 1] and each row sums to 1.
                The row i corresponds to the prediction of the ith data sample, and
                the column j to the jth class. So element [i, j] is P(y_i=k | x_i, W)
        """
        exp = np.exp(np.dot(data, w))
        return exp / np.sum(exp, axis=1)[:, np.newaxis]

    def loss_logistic_multi(self, data, labels, w):
        """
        Loss function for multi class logistic regression, i.e., multi-class entropy.

        Args:
            data (array): Input data of shape (N, D)
            labels (array): Labels of shape  (N, C)  (in one-hot representation)
            w (array): Weights of shape (D, C)
        Returns:
            float: Loss value
        """

        return -np.sum(np.log(self.f_softmax(data, w)) * labels)

    def gradient_logistic_multi(self, data, labels, w):
        """
        Compute the gradient of the entropy for multi-class logistic regression.

        Args:
            data (array): Input data of shape (N, D)
            labels (array): Labels of shape  (N, C)  (in one-hot representation)
            w (array): Weights of shape (D, C)
        Returns:
            grad (np.array): Gradients of shape (D, C)
        """

        return data.T @ (self.f_softmax(data, w) - labels)

    def logistic_regression_predict_multi(self, data, W):
        """
        Prediction the label of data for multi-class logistic regression.

        Args:
            data (array): Dataset of shape (N, D).
            W (array): Weights of multi-class logistic regression model of shape (D, C)
        Returns:
            array of shape (N,): Label predictions of data.
        """

        return np.argmax(self.f_softmax(data, W), axis=1)

    def logistic_regression_train_multi(self, data, labels):
        """
        Training function for multi class logistic regression.

        Args:
            data (array): Dataset of shape (N, D).
            labels (array): Labels of shape (N, C)
            max_iters (int): Maximum number of iterations. Default: 10
            lr (int): The learning rate of  the gradient step. Default: 0.001
            print_period (int): Number of iterations to print current loss.
                If 0, never printed.
            plot_period (int): Number of iterations to plot current predictions.
                If 0, never plotted.
        Returns:
            weights (array): weights of the logistic regression model, of shape(D, C)
        """
        D = data.shape[1]  # number of features
        C = labels.shape[1]  # number of classes
        # Random initialization of the weights
        weights = np.random.normal(0, 0.1, (D, C))
        for it in range(self.max_iters):
            gradient = self.gradient_logistic_multi(data, labels, weights)
            weights = weights - self.lr * gradient

            predictions = self.logistic_regression_predict_multi(data, weights)
            if accuracy_fn(predictions, onehot_to_label(labels)) == 100:
                break

        return weights

    def fit(self, training_data, training_labels):
        """
        Trains the model, returns predicted labels for training data.

        Arguments:
            training_data (array): training data of shape (N,D)
            training_labels (array): regression target of shape (N,)
        Returns:
            pred_labels (array): target of shape (N,)
        """

        training_labels_onehot = label_to_onehot(training_labels, get_n_classes(training_labels))
        self.weights_multi = self.logistic_regression_train_multi(training_data, training_labels_onehot)
        return self.predict(training_data)

    def predict(self, test_data):
        """
        Runs prediction on the test data.
        
        Arguments:
            test_data (array): test data of shape (N,D)
        Returns:
            pred_labels (array): labels of shape (N,)
        """

        pred_labels = self.logistic_regression_predict_multi(test_data, self.weights_multi)
        return pred_labels
