import numpy as np

import torch
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

def wrap_data(xb, yb, device):
    xb, yb = Variable(xb), Variable(yb)
    if str(device) != 'cpu':
        xb, yb = xb.cuda(), yb.cuda()

    return xb, yb

def wrap_data_with_warp(xb, yb, warpb, device):
    xb, yb, warpb = Variable(xb), Variable(yb), Variable(warpb)
    if str(device) != 'cpu':
        xb, yb, warpb = xb.cuda(), yb.cuda(), warpb.cuda()

    return xb, yb, warpb

class Solver(object):
    default_adam_args = {"lr": 1e-4,
                         "betas": (0.9, 0.999),
                         "eps": 1e-8,
                         "weight_decay": 0.0}

    def __init__(self,
                 optim=torch.optim.Adam,
                 optim_args={},
                 extra_args={},
                 loss_func=torch.nn.CrossEntropyLoss(),
                 log_dir=None):
        optim_args_merged = self.default_adam_args.copy()
        optim_args_merged.update(optim_args)
        self.optim_args = optim_args_merged
        self.optim = optim
        self.loss_func = loss_func

        self.writer = SummaryWriter(log_dir)
        self.hparam_dict = {'loss function': type(self.loss_func).__name__,
                            'optimizer': self.optim.__name__,
                            'learning rate': self.optim_args['lr'],
                            'weight_decay': self.optim_args['weight_decay'],
                            **extra_args}

        print("Hyperparameters of this solver: {}".format(self.hparam_dict))

        self._reset_histories()

    def _reset_histories(self):
        """
        Resets train and val histories for the accuracy and the loss.
        """
        self.train_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        self.val_loss_history = []

    def forward_pass(self, model, sample, device):
        xb = sample["image"]
        yb = sample["label"]
        warpb = sample.get("warp") # use get function here to ensure None is returned if key is not set instead of KeyError: https://stackoverflow.com/questions/6130768/return-none-if-dictionary-key-is-not-available

        if warpb is not None:
            xb, yb, warpb = wrap_data_with_warp(xb, yb, warpb, device)
            x = {"image": xb, "warp": warpb} # by convention this is the input to the model when optical_flow is available
            scores = model(x)
        else:
            xb, yb = wrap_data(xb, yb, device)
            scores = model(xb)

        loss = self.loss_func(scores, yb)
        acc = self.accuracy(scores, yb)

        return loss, acc

    def test(self, model, test_loader, test_prefix='/', log_nth=0):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        if test_prefix == None:
            test_prefix='/'
        elif not test_prefix.endswith('/'):
            test_prefix += '/'

        with torch.no_grad():
            test_losses = []
            test_accs = []
            for i, sample in enumerate(test_loader):
                loss, test_acc = self.forward_pass(model, sample, device)
                loss = loss.data.cpu().numpy()
                test_losses.append(loss)
                test_accs.append(test_acc)

                self.writer.add_scalar('Test/' + test_prefix + 'Batch/Loss', loss, i)
                self.writer.add_scalar('Test/' + test_prefix + 'Batch/Accuracy', test_acc, i)
                self.writer.flush()

                # Print loss every log_nth iteration
                if (i % log_nth == 0):
                    print("[Iteration {cur}/{max}] TEST loss: {loss}".format(cur=i + 1,
                                                                              max=len(test_loader),
                                                                              loss=loss))

            mean_loss = np.mean(test_losses)
            mean_acc = np.mean(test_accs)

            self.writer.add_scalar('Test/' + test_prefix + 'Mean/Loss', mean_loss, 0)
            self.writer.add_scalar('Test/' + test_prefix + 'Mean/Accuracy', mean_acc, 0)
            self.writer.flush()

            print("[TEST] mean acc/loss: {acc}/{loss}".format(acc=mean_acc, loss=mean_loss))

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
        for epoch in range(num_epochs):  # for every epoch...
            model.train()  # TRAINING mode (for dropout, batchnorm, etc.)
            train_losses = []
            train_accs = []
            for i, sample in enumerate(train_loader):  # for every minibatch in training set
                # FORWARD PASS --> Loss + acc calculation
                train_loss, train_acc = self.forward_pass(model, sample, device)

                # BACKWARD PASS --> Gradient-Descent update
                train_loss.backward()
                optim.step()
                optim.zero_grad()

                # LOGGING of loss and accuracy
                train_loss = train_loss.data.cpu().numpy()
                train_losses.append(train_loss)
                train_accs.append(train_acc)

                self.writer.add_scalar('Batch/Loss/Train', train_loss, i + epoch * iter_per_epoch)
                self.writer.add_scalar('Batch/Accuracy/Train', train_acc, i + epoch*iter_per_epoch)
                self.writer.flush()

                # Print loss every log_nth iteration
                if (i % log_nth == 0):
                    print("[Iteration {cur}/{max}] TRAIN loss: {loss}".format(cur=i + 1,
                                                                              max=iter_per_epoch,
                                                                              loss=train_loss))

            # ONE EPOCH PASSED --> calculate + log mean train accuracy/loss for this epoch
            mean_train_loss = np.mean(train_losses)
            mean_train_acc = np.mean(train_accs)

            self.train_loss_history.append(mean_train_loss)
            self.train_acc_history.append(mean_train_acc)

            self.writer.add_scalar('Epoch/Loss/Train', mean_train_loss, epoch)
            self.writer.add_scalar('Epoch/Accuracy/Train', mean_train_acc, epoch)

            print("[EPOCH {cur}/{max}] TRAIN mean acc/loss: {acc}/{loss}".format(cur=epoch + 1,
                                                                                 max=num_epochs,
                                                                                 acc=mean_train_acc,
                                                                                 loss=mean_train_loss))

            # ONE EPOCH PASSED --> calculate + log validation accuracy/loss for this epoch
            model.eval()  # EVAL mode (for dropout, batchnorm, etc.)
            with torch.no_grad():
                val_losses = []
                val_accs = []
                for i, sample in enumerate(val_loader):
                    # FORWARD PASS --> Loss + acc calculation
                    val_loss, val_acc = self.forward_pass(model, sample, device)
                    val_loss = val_loss.data.cpu().numpy()
                    val_losses.append(val_loss)
                    val_accs.append(val_acc)

                    self.writer.add_scalar('Batch/Loss/Val', val_loss, i + epoch*len(val_loader))
                    self.writer.add_scalar('Batch/Accuracy/Val', val_acc, i + epoch*len(val_loader))
                    self.writer.flush()

                    # Print loss every log_nth iteration
                    if (i % log_nth == 0):
                        print("[Iteration {cur}/{max}] Val loss: {loss}".format(cur=i + 1,
                                                                                max=len(val_loader),
                                                                                loss=val_loss))

                mean_val_loss = np.mean(val_losses)
                mean_val_acc = np.mean(val_accs)

                self.val_loss_history.append(mean_val_loss)
                self.val_acc_history.append(mean_val_acc)

                self.writer.add_scalar('Epoch/Loss/Val', mean_val_loss, epoch)
                self.writer.add_scalar('Epoch/Accuracy/Val', mean_val_acc, epoch)
                self.writer.flush()

                print("[EPOCH {cur}/{max}] VAL mean acc/loss: {acc}/{loss}".format(cur=epoch + 1,
                                                                                   max=num_epochs,
                                                                                   acc=mean_val_acc,
                                                                                   loss=mean_val_loss))

        self.writer.add_hparams(self.hparam_dict, {
            'HParam/Accuracy/Val': self.val_acc_history[-1],
            'HParam/Accuracy/Train': self.train_acc_history[-1],
            'HParam/Loss/Val': self.val_loss_history[-1],
            'HParam/Loss/Train': self.train_loss_history[-1]
        })
        self.writer.flush()
        print('FINISH.')

    def accuracy(self, scores, y):
        with torch.no_grad():
            _, preds = torch.max(scores, 1) # select highest value as the predicted class
            y_mask = y >= 0 # do not allow "-1" segmentation value
            acc = np.mean((preds == y)[y_mask].data.cpu().numpy())  # check if prediction is correct + average of it for all N inputs
            return acc
