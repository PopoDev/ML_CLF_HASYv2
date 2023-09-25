import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from src.utils import accuracy


"""
python main.py --data dataset  --method nn --nn_type mlp --lr 1e-5 --max_iters 100
n1 = 512, n2 = 256
Train set: accuracy = 5.421% - F1-score = 0.029009
Test set:  accuracy = 6.113% - F1-score = 0.030511

python main.py --data dataset  --method nn --nn_type mlp --lr 1e-4 --max_iters 100
n1 = 512, n2 = 256
Train set: accuracy = 22.100% - F1-score = 0.154465
Test set:  accuracy = 22.607% - F1-score = 0.163317

python main.py --data dataset  --method nn --nn_type mlp --lr 1e-3 --max_iters 100
n1 = 512, n2 = 256
Train set: accuracy = 71.190% - F1-score = 0.642654
Test set:  accuracy = 70.588% - F1-score = 0.633165

python main.py --data dataset  --method nn --nn_type mlp --lr 1e-2 --max_iters 100
n1 = 1024, n2 = 512
Train set: accuracy = 100.000% - F1-score = 1.000000
Test set:  accuracy = 90.657% - F1-score = 0.899648

python main.py --data dataset  --method nn --nn_type mlp --lr 1e-2 --max_iters 100
n1 = 1024, n2 = 512, n3 = 256
Train set: accuracy = 100.000% - F1-score = 1.000000
Validation set:  accuracy = 87.659% - F1-score = 0.872662
Test set:  accuracy = 89.965% - F1-score = 0.892299

n1 = 2048, n2 = 512, n3 = 128
Train set: accuracy = 100.000% - F1-score = 1.000000
Validation set:  accuracy = 88.927% - F1-score = 0.881676
Test set:  accuracy = 89.158% - F1-score = 0.881972

n1= 1536, n2=512, n3= 128
Train set: accuracy = 100.000% - F1-score = 1.000000
Validation set:  accuracy = 88.351% - F1-score = 0.879161
=============================================================
==== Test set:  accuracy = 90.311% - F1-score = 0.897573 ====
=============================================================

python main.py --data dataset  --method nn --nn_type mlp --lr 5e-3 --max_iters 100
n1= 1540, n2=512, n3= 64
Train set: accuracy = 99.204% - F1-score = 0.990928
Validation set:  accuracy = 88.812% - F1-score = 0.871173

n1= 1540, n2=742, n3= 128
Train set: accuracy = 99.166% - F1-score = 0.990572
Validation set:  accuracy = 89.389% - F1-score = 0.884215

n1= 1536, n2=512, n3= 128
Train set: accuracy = 99.204% - F1-score = 0.990741
Validation set:  accuracy = 89.735% - F1-score = 0.889666
Test set:  accuracy = 89.273% - F1-score = 0.885110

==========
== Adam ==
==========

python main.py --data dataset  --method nn --nn_type mlp --lr 1e-3 --max_iters 100
n1 = 1024, n2 = 512
Train set: accuracy = 100.000% - F1-score = 1.000000
Validation set:  accuracy = 91.811% - F1-score = 0.912584
Test set:  accuracy = 91.465% - F1-score = 0.908364

python main.py --data dataset  --method nn --nn_type mlp --lr 1e-3 --max_iters 50
Train set: accuracy = 100.000% - F1-score = 1.000000
Validation set:  accuracy = 92.503% - F1-score = 0.917466
Test set:  accuracy = 91.465% - F1-score = 0.908364

n1 = 1024, n2 = 512, n3 = 256
Train set: accuracy = 100.000% - F1-score = 1.000000
Validation set:  accuracy = 89.735% - F1-score = 0.886851
Test set:  accuracy = 90.542% - F1-score = 0.898687
"""


class MLP(nn.Module):
    """
    An MLP network which does classification.

    It should not use any convolutional layers.
    """

    DEFAULT_NUM_NEURONS = (512, 256)

    def __init__(self, input_size, n_classes, num_neurons=DEFAULT_NUM_NEURONS):
        """
        Initialize the network.
        
        You can add arguments if you want, but WITH a default value, e.g.:
            __init__(self, input_size, n_classes, my_arg=32)
        
        Arguments:
            input_size (int): size of the input
            n_classes (int): number of classes to predict
        """
        super().__init__()

        num_layers = len(num_neurons)
        layers = []

        inp = [input_size] + list(num_neurons)
        # print("inp", inp)
        out = list(num_neurons) + [n_classes]
        # print("out", out)

        for i in range(num_layers+1):
            layers.append(nn.Linear(inp[i], out[i]))
            if not i == num_layers:
                layers.append(nn.ReLU())

        # print("Layers", layers)

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        """
        Predict the class of a batch of samples with the model.

        Arguments:
            x (tensor): input batch of shape (N, D)
        Returns:
            preds (tensor): logits of predictions of shape (N, C)
                Reminder: logits are value pre-softmax.
        """
        preds = self.layers(x)

        return preds


"""
python main.py --data dataset  --method nn --nn_type cnn --lr 1e-2 --max_iters 100
Train set: accuracy = 100.000% - F1-score = 1.000000
Validation set:  accuracy = 93.195% - F1-score = 0.928416
Test set:  accuracy = 94.925% - F1-score = 0.948130

python main.py --data dataset  --method nn --nn_type cnn --lr 1e-3 --max_iters 100
Train set: accuracy = 100.000% - F1-score = 1.000000
Validation set:  accuracy = 96.655% - F1-score = 0.962209
Test set:  accuracy = 97.347% - F1-score = 0.972059
"""


class CNN(nn.Module):
    """
    A CNN which does classification.

    It should use at least one convolutional layer.
    """

    DEFAULT_LINEAR_NEURONS = [128]

    def __init__(self, input_channels, n_classes, filters=(16, 32, 64), linear_neurons=DEFAULT_LINEAR_NEURONS):
        """
        Initialize the network.
        
        You can add arguments if you want, but WITH a default value, e.g.:
            __init__(self, input_channels, n_classes, my_arg=32)
        
        Arguments:
            input_channels (int): number of channels in the input
            n_classes (int): number of classes to predict
        """
        super().__init__()

        dimensions = 32
        kernel = 3
        padding = int((kernel-1) / 2)

        num_conv_layers = len(filters)
        conv_layers = []

        channels = [input_channels] + list(filters)

        for i in range(num_conv_layers):
            conv_layers.append(nn.Conv2d(channels[i], channels[i+1], kernel_size=kernel, padding=padding))
            conv_layers.append(nn.ReLU())
            conv_layers.append(nn.MaxPool2d(kernel_size=2))
            dimensions //= 2

        self.conv_layer = nn.Sequential(*conv_layers)

        num_linear_layers = len(linear_neurons)
        linear_layers = []

        inp = [dimensions * dimensions * filters[-1]] + list(linear_neurons)
        out = list(linear_neurons) + [n_classes]

        for i in range(num_linear_layers+1):
            linear_layers.append(nn.Linear(inp[i], out[i]))
            if not i == num_linear_layers:
                linear_layers.append(nn.ReLU())

        self.linear_layer = nn.Sequential(*linear_layers)

    def forward(self, x):
        """
        Predict the class of a batch of samples with the model.

        Arguments:
            x (tensor): input batch of shape (N, Ch, H, W)
        Returns:
            preds (tensor): logits of predictions of shape (N, C)
                Reminder: logits are value pre-softmax.
        """

        x = self.conv_layer(x)
        x = x.reshape((x.shape[0], -1))
        preds = self.linear_layer(x)

        return preds


class Trainer(object):
    """
    Trainer class for the deep networks.

    It will also serve as an interface between numpy and pytorch.
    """

    def __init__(self, model, lr, epochs, batch_size):
        """
        Initialize the trainer object for a given model.

        Arguments:
            model (nn.Module): the model to train
            lr (float): learning rate for the optimizer
            epochs (int): number of epochs of training
            batch_size (int): number of data points in each batch
        """
        self.lr = lr
        self.epochs = epochs
        self.model = model
        self.batch_size = batch_size

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)  ### WRITE YOUR CODE HERE

    def train_all(self, dataloader):
        """
        Fully train the model over the epochs. 
        
        In each epoch, it calls the functions "train_one_epoch". If you want to
        add something else at each epoch, you can do it here.

        Arguments:
            dataloader (DataLoader): dataloader for training data
        """
        for ep in range(self.epochs):
            print('\n================== Epoch {}/{} =================='.format(ep + 1, self.epochs))
            self.train_one_epoch(dataloader)

            ### WRITE YOUR CODE HERE if you want to do add else at each epoch

    def train_one_epoch(self, dataloader):
        """
        Train the model for ONE epoch.

        Should loop over the batches in the dataloader. (Recall the exercise session!)
        Don't forget to set your model to training mode, i.e., self.model.train()!

        Arguments:
            dataloader (DataLoader): dataloader for training data
        """

        # Training
        self.model.train()
        for it, batch in enumerate(dataloader):
            # 5.1 Load a batch, break it down in images and targets.
            x, y = batch
            y = y.type(torch.LongTensor)

            # 5.2 Run forward pass.
            logits = self.model(x)

            # 5.3 Compute loss (using 'criterion').
            loss = self.criterion(logits, y)

            # 5.4 Run backward pass.
            loss.backward()

            # 5.5 Update the weights using 'optimizer'.
            self.optimizer.step()

            # 5.6 Zero-out the accumulated gradients.
            self.optimizer.zero_grad()

            print('\rIt {}/{}: loss train: {:.2f}, accuracy train: {:.2f}'.
                  format(it + 1, len(dataloader), loss, accuracy(logits, y)), end='')

    def predict_torch(self, dataloader):
        """
        Predict the validation/test dataloader labels using the model.

        Hints:
            1. Don't forget to set your model to eval mode, i.e., self.model.eval()!
            2. You can use torch.no_grad() to turn off gradient computation, 
            which can save memory and speed up computation. Simply write:
                with torch.no_grad():
                    # Write your code here.

        Arguments:
            dataloader (DataLoader): dataloader for validation/test data
        Returns:
            pred_labels (torch.tensor): predicted labels of shape (N,),
                with N the number of data points in the validation/test data.
        """

        # Validation
        self.model.eval()

        # Create an empty list to store the predicted labels
        labels = []

        with torch.no_grad():
            for it, batch in enumerate(dataloader):
                # Get batch of data.
                x = batch[0]

                logits = self.model(x)

                # Use torch.max to get the predicted labels
                pred = torch.max(logits, dim=1)[1]

                # Append the predicted labels to the list
                labels.append(pred)

        pred_labels = torch.cat(labels)
        return pred_labels
    
    def fit(self, training_data, training_labels):
        """
        Trains the model, returns predicted labels for training data.

        This serves as an interface between numpy and pytorch.

        Arguments:
            training_data (array): training data of shape (N,D)
            training_labels (array): regression target of shape (N,)
        Returns:
            pred_labels (array): target of shape (N,)
        """
        # First, prepare data for pytorch
        train_dataset = TensorDataset(torch.from_numpy(training_data).float(), 
                                      torch.from_numpy(training_labels))
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        self.train_all(train_dataloader)

        return self.predict(training_data)

    def predict(self, test_data):
        """
        Runs prediction on the test data.

        This serves as an interface between numpy and pytorch.
        
        Arguments:
            test_data (array): test data of shape (N,D)
        Returns:
            pred_labels (array): labels of shape (N,)
        """
        # First, prepare data for pytorch
        test_dataset = TensorDataset(torch.from_numpy(test_data).float())
        test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        pred_labels = self.predict_torch(test_dataloader)

        # We return the labels after transforming them into numpy array.
        return pred_labels.numpy()