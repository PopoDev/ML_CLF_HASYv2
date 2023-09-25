import numpy as np
import matplotlib.pyplot as plt

# Generaly utilies
##################

def label_to_onehot(labels, C=None):
    """
    Transform the labels into one-hot representations.

    Arguments:
        labels (array): labels as class indices, of shape (N,)
        C (int): total number of classes. Optional, if not given
                 it will be inferred from labels.
    Returns:
        one_hot_labels (array): one-hot encoding of the labels, of shape (N,C)
    """
    N = labels.shape[0]
    if C is None:
        C = get_n_classes(labels)
    one_hot_labels = np.zeros([N, C])
    one_hot_labels[np.arange(N), labels.astype(int)] = 1
    return one_hot_labels

def onehot_to_label(onehot):
    """
    Transform the labels from one-hot to class index.

    Arguments:
        onehot (array): one-hot encoding of the labels, of shape (N,C)
    Returns:
        (array): labels as class indices, of shape (N,)
    """
    return np.argmax(onehot, axis=1)

def append_bias_term(data):
    """
    Append to the data a bias term equal to 1.

    Arguments:
        data (array): of shape (N,D)
    Returns:
        (array): shape (N,D+1)
    """
    N = data.shape[0]
    data = np.concatenate([np.ones([N, 1]),data], axis=1)
    return data

def normalize_fn(data, means, stds):
    """
    Return the normalized data, based on precomputed means and stds.
    
    Arguments:
        data (array): of shape (N,D)
        means (array): of shape (1,D)
        stds (array): of shape (1,D)
    Returns:
        (array): shape (N,D)
    """
    # return the normalized features
    return (data - means) / stds

def get_n_classes(labels):
    """
    Return the number of classes present in the data labels.
    
    This is approximated by taking the maximum label + 1 (as we count from 0).
    """
    return int(np.max(labels) + 1)


# Metrics
#########

def accuracy_fn(pred_labels, gt_labels):
    """
    Return the accuracy of the predicted labels.
    """
    return np.mean(pred_labels == gt_labels) * 100.

def macrof1_fn(pred_labels, gt_labels):
    """Return the macro F1-score."""
    class_ids = np.unique(gt_labels)
    macrof1 = 0
    for val in class_ids:
        predpos = (pred_labels == val)
        gtpos = (gt_labels==val)
        
        tp = sum(predpos*gtpos)
        fp = sum(predpos*~gtpos)
        fn = sum(~predpos*gtpos)
        if tp == 0:
            continue
        else:
            precision = tp/(tp+fp)
            recall = tp/(tp+fn)

        macrof1 += 2*(precision*recall)/(precision+recall)

    return macrof1/len(class_ids)

def accuracy(x, y):
    """ Accuracy.

    Args:
        x (torch.Tensor of float32): Predictions (logits), shape (B, C), B is
            batch size, C is num classes.
        y (torch.Tensor of int64): GT labels, shape (B, ),
            B = {b: b \in {0 .. C-1}}.

    Returns:
        Accuracy, in [0, 1].
    """
    x = x.detach().cpu().numpy()
    y = y.detach().cpu().numpy()
    return np.mean(np.argmax(x, axis=1) == y)

# Visualisation
###############

def labels_name():
    return ["\\int", "\\sum", "\\infty", "\\alpha", "\\xi", "\\equiv", "\\partial", "\\mathds{R}", "\\in", "\\square",
            "\\forall", "\\approx", "\\sim", "\\Rightarrow", "\\subseteq", "\\pi", "\\pm", "\\neq", "\\varphi", "\\times"]


def label_name(label):
    return labels_name()[label]


def display_symbol(symbol, label_gt, label_pred, ax=None):
    """Graphically display a 32x32 image representing a symbol, and optionally show the corresponding label."""
    if ax is None:
        plt.figure()
        fig = plt.imshow(symbol.reshape(32, 32))
    else:
        fig = ax.imshow(symbol.reshape(32, 32))
    fig.set_cmap('gray_r')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    fig.axes.set_title(f"GT label: {label_gt}", fontsize=12)
    fig.axes.annotate(f"Inferred label: {label_pred}", xy=(0.5, -0.20), xycoords='axes fraction',
                      fontsize=12, ha='center', color='green' if label_gt == label_pred else 'red')


def visualize_predictions(data, labels_gt, labels_pred):
    """ Visualizes the dataset labels.

    Args:
        data (np.array): Dataset, shape (N, D).
        labels_gt (np.array): GT labels, shape (N, ).
        labels_pred (np.array): Predicted labels, shape (N, )
    """

    # Randomly select 10 points in the data set
    number = 10
    rand_idx = np.random.permutation(data.shape[0])[:number]
    samples = data[rand_idx]
    labels_gt_random = labels_gt[rand_idx]
    labels_pred_random = labels_pred[rand_idx]

    fig, axes = plt.subplots(3, 5, figsize=(15, 6), gridspec_kw={'height_ratios': [1, 1, 5]})
    fig.suptitle('Result of classification using SVM RBF with 10 examples of images and their labels', fontsize=16)
    fig.subplots_adjust(hspace=0.4, wspace=0.25)

    for i in range(number):
        display_symbol(samples[i], label_name(labels_gt_random[i]), label_name(labels_pred_random[i]), ax=axes.flat[i])

    # Histogram
    ax_data = plt.subplot2grid((3, 5), (2, 0), rowspan=1, colspan=5)
    ax_data.set_xlabel('GT Labels')
    ax_data.set_ylabel('Inferred Labels')

    number_symbols = get_n_classes(labels_gt)
    bottom = np.zeros(number_symbols)

    for i in range(number_symbols):
        classification = np.bincount(labels_pred[labels_gt == i], minlength=number_symbols)
        ax_data.bar(labels_name(), classification, bottom=bottom, label=label_name(i))
        print(classification, classification.size)

        labels = [classification[v] if v == i else '' for v in range(number_symbols)]
        ax_data.bar_label(ax_data.containers[i], labels=labels, label_type='center')

        bottom += np.array(classification)

    # Set an offset that is used to bump the label up a bit above the bar.
    y_offset = 2
    # Add total count to each bar.
    for i, total in enumerate(bottom):
        ax_data.text(label_name(i), total + y_offset, round(total), ha='center', weight='bold')

    ax_data.set_ylim(top=np.max(bottom) + 10)
    ax_data.legend(loc='upper right')
    fig.tight_layout()
    plt.savefig('result.png')
