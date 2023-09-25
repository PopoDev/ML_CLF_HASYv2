import time

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import figaspect
from sklearn.utils import shuffle
from torchinfo import summary

from src.data import load_data
from src.methods.deep_network import MLP, Trainer, CNN
from src.methods.kmeans import KMeans
from src.methods.logistic_regression import LogisticRegression
from src.methods.pca import PCA
from src.methods.svm import SVM
from src.utils import normalize_fn, accuracy_fn, macrof1_fn, get_n_classes


def mlp_validation(xtrain, ytrain, xvalid, yvalid):
    architectures = {
        2: [[16, 32], [64, 32], [256, 128], [512, 256], [1024, 512], [1536, 512]],
        3: [[64, 32, 16], [128, 64, 32], [256, 128, 64], [512, 256, 128], [1024, 512, 128], [1536, 512, 128]]
    }

    total_architectures = 0
    input_size = 1024
    n_classes = 20
    lr = 1e-3
    max_iters = 100
    batch_size = 64

    w, h = figaspect(1 / 3)
    plt.figure(figsize=(w, h))
    fig, ax1 = plt.subplots()
    fig.suptitle("MLP Performance on the validation set for different architectures")

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax1.set_xlim([0, 3_000_000])
    ax1.set_xlabel("Number of weights (Neurons)")
    ax1.set_ylabel("Performance (accuracy)")
    ax2.set_ylabel('Performance (time)')
    ax2.set_ylim([0, 150])

    for num_layer, layers in architectures.items():
        weights = []
        accuracies = []
        runtimes = []
        total_architectures += len(layers)

        for layer in layers:
            model = MLP(input_size, n_classes, layer)

            summary(model)

            start_time = time.time()
            method_obj = Trainer(model, lr, max_iters, batch_size)
            preds_train = method_obj.fit(xtrain, ytrain)
            preds = method_obj.predict(xvalid)
            end_time = time.time()

            acc = accuracy_fn(preds_train, ytrain)
            macrof1 = macrof1_fn(preds_train, ytrain)
            print(f"\nTrain set: accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")

            acc = accuracy_fn(preds, yvalid)
            macrof1 = macrof1_fn(preds, yvalid)
            print(f"Validation set:  accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}\n")

            pytorch_total_params = sum(p.numel() for p in model.parameters())
            weights.append(pytorch_total_params)
            accuracies.append(acc)
            runtimes.append(end_time - start_time)

        line, = ax1.plot(weights, accuracies, linestyle='-', label=f'{num_layer} Hidden layers', marker='o')
        ax2.plot(weights, runtimes, linestyle='--', label=f'[{num_layer}] Runtime', color=line.get_color(), alpha=.5)
        for i, xy in enumerate(zip(weights, accuracies)):
            ax1.annotate(layers[i], xy=xy, textcoords='data')

    ax1.legend(loc='lower right')
    ax2.legend(loc='center right')
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(f"mlp_validation_iters={max_iters}_lr={lr}_arch={total_architectures}_time.png")


def cnn_validation(xtrain, ytrain, xvalid, yvalid):
    architectures = {
        2: [[1, 2], [2, 4], [4, 8], [8, 16], [16, 32]],
        3: [[2, 4, 8], [4, 8, 16], [8, 16, 32], [16, 32, 64], [32, 64, 96]]
    }

    total_architectures = 0
    input_size = 1024
    n_classes = 20
    lr = 1e-3
    max_iters = 25
    batch_size = 64

    w, h = figaspect(1 / 3)
    plt.figure(figsize=(w, h))
    fig, ax1 = plt.subplots()
    fig.suptitle("CNN Performance on the validation set for different architectures using Adam")

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax1.set_xlim([0, 300_000])
    ax1.set_xlabel("Number of weights")
    ax1.set_ylabel("Accuracy")
    ax2.set_ylabel('Time')
    ax2.set_ylim([0, 150])

    for num_layer, layers in architectures.items():
        weights = []
        accuracies = []
        runtimes = []
        total_architectures += len(layers)

        for layer in layers:
            input_channels = 1
            xtrain = xtrain.reshape((n_train - n_val, input_channels, 32, 32))
            xvalid = xvalid.reshape((n_val, input_channels, 32, 32))
            model = CNN(input_channels=input_channels, n_classes=n_classes, filters=layer)
            summary(model)

            start_time = time.time()
            method_obj = Trainer(model, lr, max_iters, batch_size)
            preds_train = method_obj.fit(xtrain, ytrain)
            preds = method_obj.predict(xvalid)
            end_time = time.time()

            acc = accuracy_fn(preds_train, ytrain)
            macrof1 = macrof1_fn(preds_train, ytrain)
            print(f"\nTrain set: accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")

            acc = accuracy_fn(preds, yvalid)
            macrof1 = macrof1_fn(preds, yvalid)
            print(f"Validation set:  accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}\n")

            pytorch_total_params = sum(p.numel() for p in model.parameters())
            weights.append(pytorch_total_params)
            accuracies.append(acc)
            runtimes.append(end_time - start_time)

        line, = ax1.plot(weights, accuracies, linestyle='-', label=f'{num_layer} Convolution layers and 1 linear layer with 128 neurons', marker='o')
        ax2.plot(weights, runtimes, linestyle='--', label=f'[{num_layer}] Runtime', color=line.get_color(), alpha=.5)
        for i, xy in enumerate(zip(weights, accuracies)):
            ax1.annotate(layers[i], xy=xy, textcoords='data')

    ax1.legend(loc='lower right')
    ax2.legend(loc='center right')
    plt.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(f"cnn_validation_iters={max_iters}_lr={lr}_arch={total_architectures}_time.png", bbox_inches='tight')


def pca_validation(xtrain, ytrain, xvalid, yvalid):

    w, h = figaspect(1 / 3)
    plt.figure(figsize=(w, h))
    fig, ax1 = plt.subplots()
    #fig.suptitle("PCA Performance on the validation set with MS1 methods")
    fig.suptitle("PCA Performance on the validation set using different ML methods")

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax1.set_xlabel("Dimensions d")
    ax1.set_ylabel("Accuracy")
    ax2.set_ylabel('%Time compared to no reduction D=1024')
    ax2.set_ylim([0, 100])

    d_start = 100
    d_step = 100
    D = 1024
    methods = {'KMeans K=400': KMeans(K=400),
               'LogisticRegression': LogisticRegression(lr=0.0001, max_iters=200),
               'SVM RBF kernel': SVM(C=1000, kernel='rbf', gamma=0.0005)
               }

    dimensions = range(d_start, D, d_step)

    for i, (method_name, method_obj) in enumerate(methods.items()):
        print("================ Without reduction D=1024 =================")
        start_time = time.time()
        preds_train = method_obj.fit(xtrain, ytrain)
        preds = method_obj.predict(xvalid)
        end_time = time.time()

        ## Report results: performance on train and valid/test sets
        acc = accuracy_fn(preds_train, ytrain)
        macrof1 = macrof1_fn(preds_train, ytrain)
        print(f"\nTrain set: accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")

        acc = accuracy_fn(preds, yvalid)
        macrof1 = macrof1_fn(preds, yvalid)
        print(f"Validation set:  accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")

        D_runtime = end_time - start_time
        D_acc = acc

        accuracies = []
        runtimes = []

        for d in dimensions:
            print(f"============ Dimensions={d} ===========")
            pca_obj = PCA(d)
            exvar = pca_obj.find_principal_components(xtrain)
            xtrain_reduce = pca_obj.reduce_dimension(xtrain)
            xvalid_reduce = pca_obj.reduce_dimension(xvalid)

            print(f"[PCA] xtrain: {xtrain_reduce.shape}, xvalid: {xvalid_reduce.shape}")
            print(f"[PCA] exvar = {exvar}")

            ## 4. Train and evaluate the method

            start_time = time.time()
            preds_train = method_obj.fit(xtrain_reduce, ytrain)
            preds = method_obj.predict(xvalid_reduce)
            end_time = time.time()

            ## Report results: performance on train and valid/test sets
            acc = accuracy_fn(preds_train, ytrain)
            macrof1 = macrof1_fn(preds_train, ytrain)
            print(f"\nTrain set: accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")

            acc = accuracy_fn(preds, yvalid)
            macrof1 = macrof1_fn(preds, yvalid)
            print(f"Validation set:  accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")

            accuracies.append(acc)
            runtimes.append((end_time - start_time) * 100 / D_runtime)

        line, = ax1.plot(dimensions, accuracies, linestyle='-', label=f'[{i+1}] {method_name}')
        ax2.plot(dimensions, runtimes, linestyle='--', label=f'[{i+1}] %Runtime', color=line.get_color(), alpha=.5)

    """
    ax1.legend(loc='lower right')
    ax2.legend(loc='center right')
    plt.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(f"pca_validation_MS1", bbox_inches='tight')
    """

    # MLP
    """
    fig, ax1 = plt.subplots()
    fig.suptitle("PCA Performance on the validation set with MLP [1024, 512]")

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax1.set_xlabel("Dimensions d")
    ax1.set_ylabel("Accuracy")
    ax2.set_ylabel('Time')
    """

    n_classes = get_n_classes(ytrain)
    lr = 1e-3
    max_iters = 10
    batch_size = 64

    input_size = D
    model = MLP(input_size=input_size, n_classes=n_classes, num_neurons=[1024, 512])
    summary(model)

    start_time = time.time()
    method_obj = Trainer(model, lr, max_iters, batch_size)
    preds_train = method_obj.fit(xtrain, ytrain)
    preds = method_obj.predict(xvalid)
    end_time = time.time()

    acc = accuracy_fn(preds_train, ytrain)
    macrof1 = macrof1_fn(preds_train, ytrain)
    print(f"\nTrain set: accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")

    acc = accuracy_fn(preds, yvalid)
    macrof1 = macrof1_fn(preds, yvalid)
    print(f"Validation set:  accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}\n")
    D_runtime = end_time - start_time

    mlp_accuracies = []
    mlp_runtimes = []
    for d in dimensions:
        print(f"============ Dimensions={d} ===========")
        pca_obj = PCA(d)
        exvar = pca_obj.find_principal_components(xtrain)
        xtrain_reduce = pca_obj.reduce_dimension(xtrain)
        xvalid_reduce = pca_obj.reduce_dimension(xvalid)

        print(f"[PCA] xtrain: {xtrain_reduce.shape}, xvalid: {xvalid_reduce.shape}")
        print(f"[PCA] exvar = {exvar}")

        input_size = d
        model = MLP(input_size=input_size, n_classes=n_classes, num_neurons=[1024, 512])
        summary(model)

        start_time = time.time()
        method_obj = Trainer(model, lr, max_iters, batch_size)
        preds_train = method_obj.fit(xtrain_reduce, ytrain)
        preds = method_obj.predict(xvalid_reduce)
        end_time = time.time()

        acc = accuracy_fn(preds_train, ytrain)
        macrof1 = macrof1_fn(preds_train, ytrain)
        print(f"\nTrain set: accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")

        acc = accuracy_fn(preds, yvalid)
        macrof1 = macrof1_fn(preds, yvalid)
        print(f"Validation set:  accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}\n")

        mlp_accuracies.append(acc)
        mlp_runtimes.append((end_time - start_time) * 100 / D_runtime)

    index = len(methods) + 1
    line, = ax1.plot(dimensions, mlp_accuracies, linestyle='-', label=f'[{index}] MLP [1024, 512]')
    ax2.plot(dimensions, mlp_runtimes, linestyle='--', label=f'[{index}] %Runtime', color=line.get_color(), alpha=.5)

    ax1.legend(loc='lower right')
    ax2.legend(loc='center right')
    plt.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(f"pca_validation_{d_step}", bbox_inches='tight')


if __name__ == '__main__':
    ## 1. First, we load our data and flatten the images into vectors
    xtrain, xtest, ytrain, ytest = load_data("../../dataset_MS2")
    xtrain = xtrain.reshape(xtrain.shape[0], -1)
    xtest = xtest.reshape(xtest.shape[0], -1)

    ## 2. Then we must prepare it. This is were you can create a validation set,
    #  normalize, add bias, etc.
    xtrain, ytrain = shuffle(xtrain, ytrain)
    xtest, ytest = shuffle(xtest, ytest)

    n_train = xtrain.shape[0]
    n_test = xtest.shape[0]
    n_val = n_test

    # Make a validation set (it can overwrite xtest, ytest)
    xvalid = xtrain[:n_val]  # xvalid
    yvalid = ytrain[:n_val]

    xtrain = xtrain[n_val:]
    ytrain = ytrain[n_val:]

    ### WRITE YOUR CODE HERE to do any other data processing

    means = xtrain.mean(axis=0, keepdims=True)
    stds = xtrain.std(axis=0, keepdims=True)

    xtrain = normalize_fn(xtrain, means, stds)
    xvalid = normalize_fn(xvalid, means, stds)
    xtest = normalize_fn(xtest, means, stds)

    #mlp_validation(xtrain, ytrain, xvalid, yvalid)
    #cnn_validation(xtrain, ytrain, xvalid, yvalid)
    pca_validation(xtrain, ytrain, xvalid, yvalid)
