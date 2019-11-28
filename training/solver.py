from random import shuffle
import numpy as np

import torch
from torch.autograd import Variable

def wrap_data(xb, yb, device):
    xb, yb = Variable(xb), Variable(yb)
    if str(device) != 'cpu':
        xb, yb = xb.cuda(), yb.cuda()

    return xb, yb

class Solver(object):
    default_adam_args = {"lr": 1e-4,
                         "betas": (0.9, 0.999),
                         "eps": 1e-8,
                         "weight_decay": 0.0}

    def __init__(self, optim=torch.optim.Adam, optim_args={},
                 loss_func=torch.nn.CrossEntropyLoss()):
        optim_args_merged = self.default_adam_args.copy()
        optim_args_merged.update(optim_args)
        self.optim_args = optim_args_merged
        self.optim = optim
        self.loss_func = loss_func

        self._reset_histories()

    def _reset_histories(self):
        """
        Resets train and val histories for the accuracy and the loss.
        """
        self.train_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        self.val_loss_history = []

    def train(self, model, train_loader, val_loader, num_epochs=10, log_nth=0):
        """
        Train a given model with the provided data.

        Inputs:
        - model: model object initialized from a torch.nn.Module
        - train_loader: train data in torch.utils.data.DataLoader
        - val_loader: val data in torch.utils.data.DataLoader
        - num_epochs: total number of training epochs
        - log_nth: log training accuracy and loss every nth iteration
        """
        optim = self.optim(filter(lambda p: p.requires_grad, model.parameters()), **self.optim_args)
        self._reset_histories()
        iter_per_epoch = len(train_loader)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)

        print('START TRAIN on device: {}'.format(device))
        ########################################################################
        # TODO:                                                                #
        # Write your own personal training method for our solver. In each      #
        # epoch iter_per_epoch shuffled training batches are processed. The    #
        # loss for each batch is stored in self.train_loss_history. Every      #
        # log_nth iteration the loss is logged. After one epoch the training   #
        # accuracy of the last mini batch is logged and stored in              #
        # self.train_acc_history. We validate at the end of each epoch, log    #
        # the result and store the accuracy of the entire validation set in    #
        # self.val_acc_history.                                                #
        #                                                                      #
        # Your logging could like something like:                              #
        #   ...                                                                #
        #   [Iteration 700/4800] TRAIN loss: 1.452                             #
        #   [Iteration 800/4800] TRAIN loss: 1.409                             #
        #   [Iteration 900/4800] TRAIN loss: 1.374                             #
        #   [Epoch 1/5] TRAIN acc/loss: 0.560/1.374                            #
        #   [Epoch 1/5] VAL   acc/loss: 0.539/1.310                            #
        #   ...                                                                #
        ########################################################################

        #import time

        for epoch in range(num_epochs):  # for every epoch...
            model.train()  # TRAINING mode (for dropout, batchnorm, etc.)
            for i, sample in enumerate(train_loader):  # for every minibatch in training set
                # DATA conversion (for cuda and cpu support)
                #start = time.time()
                xb = sample["image"]
                yb = sample["label"]
                xb, yb = wrap_data(xb, yb, device)
                #end = time.time()
                #print("Converting input took {} seconds".format(end - start))

                # FORWARD PASS --> Loss calculation
                #start = time.time()
                scores = model(xb)
                loss = self.loss_func(scores, yb)
                #end = time.time()
                #print("Forward pass took {} seconds".format(end - start))

                # BACKWARD PASS --> Gradient-Descent update
                #start = time.time()
                loss.backward()
                optim.step()
                optim.zero_grad()
                loss = loss.data.cpu().numpy()
                self.train_loss_history.append(loss)
                #end = time.time()
                #print("Backward pass and loss calculation took {} seconds".format(end - start))

                # Print loss every log_nth iteration
                if (i % log_nth == 0):
                    print("[Iteration {cur}/{max}] TRAIN loss: {loss}".format(cur=i + 1, max=iter_per_epoch, loss=loss))

                # If last batch: calculate training accuracy
                if i == iter_per_epoch - 1:
                    train_acc = self.accuracy(scores, yb)
                    self.train_acc_history.append(train_acc)
                    print("[EPOCH {cur}/{max}] TRAIN acc/loss: {acc}/{loss}".format(cur=epoch + 1, max=num_epochs,
                                                                                    acc=train_acc, loss=loss))

            # ONE EPOCH PASSED --> calculate validation accuracy
            model.eval()  # EVAL mode (for dropout, batchnorm, etc.)
            with torch.no_grad():
                losses = []
                val_acc = []
                for sample in val_loader:
                    xb = sample["image"].float()
                    yb = sample["label"].long()
                    xb, yb = wrap_data(xb, yb, device)

                    # FORWARD PASS --> Loss calculation
                    scores = model(xb)
                    loss = self.loss_func(scores, yb)
                    loss = loss.data.cpu().numpy()
                    losses.append(loss)
                    val_acc.append(self.accuracy(scores, yb))

                val_loss = np.mean(losses)
                val_acc = np.mean(val_acc)

                self.val_loss_history.append(val_loss)
                self.val_acc_history.append(val_acc)
                print(
                    "[EPOCH {cur}/{max}] VAL acc/loss: {acc}/{loss}".format(cur=epoch + 1, max=num_epochs, acc=val_acc,
                                                                            loss=val_loss))
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################
        print('FINISH.')

    def accuracy(self, scores, y):
        with torch.no_grad():
            _, preds = torch.max(scores, 1) # select highest value as the predicted class
            y_mask = y >= 0 # do not allow "-1" segmentation value
            acc = np.mean((preds == y)[y_mask].data.cpu().numpy())  # check if prediction is correct + average of it for all N inputs
            return acc
