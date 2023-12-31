import argparse

from sklearn.utils import shuffle
from torchinfo import summary

from src.data import load_data
from src.methods.deep_network import MLP, CNN, Trainer
from src.methods.dummy_methods import DummyClassifier
from src.methods.kmeans import KMeans
from src.methods.logistic_regression import LogisticRegression
from src.methods.pca import PCA
from src.methods.svm import SVM
from src.utils import normalize_fn, accuracy_fn, macrof1_fn, get_n_classes, visualize_predictions


def main(args):
    """
    The main function of the script. Do not hesitate to play with it
    and add your own code, visualization, prints, etc!

    Arguments:
        args (Namespace): arguments that were parsed from the command line (see at the end
                          of this file). Their value can be accessed as "args.argument".
    """
    ## 1. First, we load our data and flatten the images into vectors
    xtrain, xtest, ytrain, ytest = load_data(args.data)
    xtrain_dim, xtest_dim = xtrain.shape, xtest.shape
    print(f"xtrain: {xtrain_dim}, xtest: {xtest_dim}")
    dimension = xtrain_dim[1]

    xtrain = xtrain.reshape(xtrain.shape[0], -1)
    xtest = xtest.reshape(xtest.shape[0], -1)

    ## 2. Then we must prepare it. This is were you can create a validation set,
    #  normalize, add bias, etc.

    xtrain, ytrain = shuffle(xtrain, ytrain)
    xtest, ytest = shuffle(xtest, ytest)

    n_total = 3505
    n_test = 867
    n_train = n_total - n_test
    n_val = n_test

    # Make a validation set
    if not args.test:
        ### WRITE YOUR CODE HERE
        xtest = xtrain[:n_val]  # xvalid overwrite xtest
        ytest = ytrain[:n_val]

        xtrain = xtrain[n_val:]
        ytrain = ytrain[n_val:]

    ### WRITE YOUR CODE HERE to do any other data processing
    means = xtrain.mean(axis=0, keepdims=True)
    stds = xtrain.std(axis=0, keepdims=True)

    xtrain = normalize_fn(xtrain, means, stds)
    xtest = normalize_fn(xtest, means, stds)

    input_size = dimension * dimension

    # Dimensionality reduction (MS2)
    if args.use_pca:
        print("Using PCA")
        pca_obj = PCA(d=args.pca_d)
        ### WRITE YOUR CODE HERE: use the PCA object to reduce the dimensionality of the data
        exvar = pca_obj.find_principal_components(xtrain)
        xtrain = pca_obj.reduce_dimension(xtrain)
        xtest = pca_obj.reduce_dimension(xtest)
        input_size = args.pca_d
        print(f"[PCA] xtrain: {xtrain_dim}, xtest: {xtest_dim}")
        print(f"[PCA] exvar = {exvar}")

    ## 3. Initialize the method you want to use.

    # Neural Networks (MS2)
    if args.method == "nn":
        print("Using deep network")

        # Prepare the model (and data) for Pytorch
        # Note: you might need to reshape the image data depending on the network you use!
        n_classes = get_n_classes(ytrain)
        if args.nn_type == "mlp":
            ### WRITE YOUR CODE HERE
            model = MLP(input_size=input_size, n_classes=n_classes, num_neurons=[1024, 512])

        elif args.nn_type == "cnn":
            ### WRITE YOUR CODE HERE
            input_channels = 1
            xtrain = xtrain.reshape((n_total if args.test else n_train, input_channels, dimension, dimension))
            xtest = xtest.reshape((n_test, input_channels, dimension, dimension))
            model = CNN(input_channels=input_channels, n_classes=n_classes, filters=(32, 64, 96))

        summary(model)

        # Trainer object
        method_obj = Trainer(model, lr=args.lr, epochs=args.max_iters, batch_size=args.nn_batch_size)

    # Follow the "DummyClassifier" example for your methods (MS1)
    elif args.method == "dummy_classifier":
        method_obj = DummyClassifier(arg1=1, arg2=2)
    elif args.method == "kmeans":  ### WRITE YOUR CODE HERE
        method_obj = KMeans(args.K)

    elif args.method == "logistic_regression":
        method_obj = LogisticRegression(args.lr, args.max_iters)

    elif args.method == "svm":
        method_obj = SVM(args.svm_c, args.svm_kernel, args.svm_gamma, args.svm_degree, args.svm_coef0)

    ## 4. Train and evaluate the method

    # Fit (:=train) the method on the training data
    preds_train = method_obj.fit(xtrain, ytrain)

    # Predict on unseen data
    preds = method_obj.predict(xtest)

    xtrain = xtrain.reshape(xtrain.shape[0], -1)
    xtest = xtest.reshape(xtest.shape[0], -1)

    ## Report results: performance on train and valid/test sets
    acc = accuracy_fn(preds_train, ytrain)
    macrof1 = macrof1_fn(preds_train, ytrain)
    print(f"\nTrain set: accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")

    acc = accuracy_fn(preds, ytest)
    macrof1 = macrof1_fn(preds, ytest)
    print(f"{'Test' if args.test else 'Validation'} set:  accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")

    ### WRITE YOUR CODE HERE if you want to add other outputs, visualization, etc.
    if not args.use_pca:
        visualize_predictions(xtest, ytest, preds)


if __name__ == '__main__':
    # Definition of the arguments that can be given through the command line (terminal).
    # If an argument is not given, it will take its default value as defined below.
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default="dataset_HASYv2", type=str, help="the path to wherever you put the data, if it's in the parent folder, you can use ../dataset_HASYv2")
    parser.add_argument('--method', default="dummy_classifier", type=str, help="dummy_classifier / kmeans / logistic_regression / svm / nn (MS2)")
    parser.add_argument('--K', type=int, default=10, help="number of clusters for K-Means")
    parser.add_argument('--lr', type=float, default=1e-5, help="learning rate for methods with learning rate")
    parser.add_argument('--max_iters', type=int, default=100, help="max iters for methods which are iterative")
    parser.add_argument('--test', action="store_true", help="train on whole training data and evaluate on the test data, otherwise use a validation set")
    parser.add_argument('--svm_c', type=float, default=1., help="Constant C in SVM method")
    parser.add_argument('--svm_kernel', default="linear", help="kernel in SVM method, can be 'linear' or 'rbf' or 'poly'(polynomial)")
    parser.add_argument('--svm_gamma', type=float, default=1., help="gamma prameter in rbf/polynomial SVM method")
    parser.add_argument('--svm_degree', type=int, default=1, help="degree in polynomial SVM method")
    parser.add_argument('--svm_coef0', type=float, default=0., help="coef0 in polynomial SVM method")

    ### WRITE YOUR CODE HERE: feel free to add more arguments here if you need!

    # MS2 arguments
    parser.add_argument('--use_pca', action="store_true", help="to enable PCA")
    parser.add_argument('--pca_d', type=int, default=200, help="output dimensionality after PCA")
    parser.add_argument('--nn_type', default="mlp", help="which network to use, can be 'mlp' or 'cnn'")
    parser.add_argument('--nn_batch_size', type=int, default=64, help="batch size for NN training")

    # "args" will keep in memory the arguments and their values,
    # which can be accessed as "args.data", for example.
    args = parser.parse_args()
    main(args)
