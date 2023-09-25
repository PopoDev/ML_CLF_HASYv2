"""
You are allowed to use the `sklearn` package for SVM.

See the documentation at https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
"""
from sklearn.svm import SVC

"""
python main.py --data dataset  --method svm --svm_c 1. --svm_kernel rbf --svm_gamma 0.01

Train set: accuracy = 100.000% - F1-score = 1.000000
Validation set:  accuracy = 93.409% - F1-score = 0.930927
Test set:  accuracy = 94.727% - F1-score = 0.944764

python main.py --data dataset --method svm --svm_c 1200 --svm_kernel rbf --svm_gamma 0.0006 --test

Train set: accuracy = 100.000% - F1-score = 1.000000
Test set:  accuracy = 96.610% - F1-score = 0.964252

python main.py --data dataset --method svm --svm_c 1000 --svm_kernel rbf --svm_gamma 0.0005 --test

Train set: accuracy = 100.000% - F1-score = 1.000000
Test set:  accuracy = 96.610% - F1-score = 0.964029
"""


class SVM(object):
    """
    SVM method.
    """

    def __init__(self, C, kernel, gamma=1., degree=1, coef0=0.):
        """
        Initialize the new object (see dummy_methods.py)
        and set its arguments.

        Arguments:
            C (float): the weight of penalty term for misclassifications
            kernel (str): kernel in SVM method, can be 'linear', 'rbf' or 'poly' (:=polynomial)
            gamma (float): gamma prameter in rbf and polynomial SVM method
            degree (int): degree in polynomial SVM method
            coef0 (float): coef0 in polynomial SVM method
        """
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.svc = SVC(C=self.C, kernel=self.kernel, degree=self.degree, gamma=self.gamma, coef0=self.coef0)
        
    def fit(self, training_data, training_labels):
        """
        Trains the model by SVM, then returns predicted labels for training data.

        Arguments:
            training_data (array): training data of shape (N,D)
            training_labels (array): regression target of shape (N,)
        Returns:
            pred_labels (array): target of shape (N,)
        """

        self.svc = self.svc.fit(training_data, training_labels)
        return self.predict(training_data)
    
    def predict(self, test_data):
        """
        Runs prediction on the test data.

        Arguments:
            test_data (array): test data of shape (N,D)
        Returns:
            pred_labels (array): labels of shape (N,)
        """

        pred_labels = self.svc.predict(test_data)
        return pred_labels
