import copy
import os
import time

import torch
from tensorboardX import SummaryWriter
from termcolor import colored

from FederatedLearning.Utilities import Utilities
from FederatedLearning.Models import MLP, CNNMnist, CNNFashionMnist
from FederatedLearning.Options import args_parser
from FederatedLearning.Update import LocalUpdate


class Federated:

    def __init__(self, agent_name, model_path, dataset, model_type):
        self.global_weights = None
        self.agent_name = agent_name
        self.model_path = model_path
        self.dataset = dataset
        self.model_type = model_type

        # Training
        self.end_time = None
        self.model = None
        self.train_loss, self.train_accuracy = [], []
        self.val_acc_list, self.net_list = [], []
        self.cv_loss, self.cv_acc = [], []
        self.print_every = 2
        self.val_loss_pre, self.counter = 0, 0

        self.start_time = time.time()
        self.best_valid_loss = 0

        # Define paths
        self.path_project = os.path.abspath('../..')
        self.logger = SummaryWriter('../logs')

        self.args = args_parser()
        print(colored('=' * 30, 'green'))
        print(self.args)
        print(colored('=' * 30, 'green'))

        self.utilities = Utilities()
        # Define the different parameters to configure the training
        if self.args.gpu:
            torch.cuda.set_device(self.args.gpu)
        self.device = 'cuda' if self.args.gpu else 'cpu'

        # Load dataset and user groups
        print(self.dataset)
        self.train_dataset, self.test_dataset, self.user_groups = self.utilities.get_dataset(self.args, self.dataset)

    def build_model(self):
        """
        Builds the agent's ML model
        """
        if self.model_type == 'cnn':
            if self.dataset == 'mnist':
                self.model = CNNMnist(args=self.args)
            elif self.dataset == 'fmnist':
                self.model = CNNFashionMnist(args=self.args)

        elif self.model_type == 'mlp':
            img_size = self.train_dataset[0][0].shape
            len_in = 1
            for x in img_size:
                len_in *= x
            self.model = MLP(dim_in=len_in, dim_hidden=32, dim_out=self.args.num_classes)

        else:
            exit('Error: unrecognized model')

        if self.model_path is not None:
            print("Using model provided in file " + self.model_path)
            self.model.load_state_dict(torch.load(self.model_path))

    def set_model(self):
        # Set the model to train and send it to device.
        self.model.to(self.device)

        # Copy weights
        self.global_weights = self.model.state_dict()

    def print_model(self):
        print(self.model)

    async def train_local_model(self, epoch=1):
        """
        Train the agent's local model
        :param epoch: number of epochs to train the model for
        :return: the weights, losses, training and test accuracies and losses
        """
        local_weights, local_losses = [], []

        local_update = LocalUpdate(args=self.args, dataset=self.train_dataset,
                                   idxs=self.user_groups[0], logger=self.logger)
        self.model, w, loss = await local_update.update_weights(
            model=copy.deepcopy(self.model))

        # Save the model locally
        torch.save(self.model.state_dict(), "Saved Models/model.pt")

        local_weights.append(copy.deepcopy(w))
        local_losses.append(copy.deepcopy(loss))

        train_acc, train_loss, test_acc, test_loss = self.get_accuracy(local_update)
        return local_weights, local_losses, train_acc, train_loss, test_acc, test_loss

    def get_accuracy(self, local_update):
        """
        Calculate the training accuracy, training loss, test accuracy and test loss
        :param local_update: the Update object related to the agent
        :return: training accuracy, training loss, test accuracy and test loss
        """
        self.model.eval()

        acc, loss = local_update.inference(model=self.model)

        print("[{}] Train Accuracy : {}%".format(self.agent_name, round(acc*100, 2)))
        print("[{}] Train Loss : {}".format(self.agent_name, round(loss, 4)))

        self.train_accuracy.append(acc)
        self.train_loss.append(loss)

        test_acc, test_loss = local_update.test_inference(self.args, self.model, self.test_dataset)

        print("[{}] Test Accuracy: {}%".format(self.agent_name, round(test_acc*100, 2)))
        print("[{}] Test Loss: {}".format(self.agent_name, round(test_loss, 4)))

        return [round(acc*100, 2), round(loss, 2), round(test_acc*100, 2), round(test_loss, 4)]
