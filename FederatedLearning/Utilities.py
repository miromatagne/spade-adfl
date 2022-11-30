from torchvision import datasets, transforms

import Config
from FederatedLearning.Sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal


class Utilities:
    def get_dataset(self, args, dataset):
        """
        Returns train and test datasets and a user group which is a dict where
        the keys are the user index and the values are the corresponding data for
        each of those users
        :param args: options
        :param dataset: dataset the model should be trained and tested on
        :return: train and test datasets, and user groups
        """
        train_dataset = ""
        test_dataset = ""
        user_groups = ""

        if dataset == 'mnist' or 'fmnist':
            if dataset == 'mnist':
                data_dir = Config.data_set_path + '/mnist/'
            else:
                data_dir = Config.data_set_path + '/fmnist/'

            apply_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))])

            if dataset == 'mnist':
                train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                               transform=apply_transform)

                test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                              transform=apply_transform)

            else:
                train_dataset = datasets.FashionMNIST(data_dir, train=True, download=True,
                                                      transform=apply_transform)

                test_dataset = datasets.FashionMNIST(data_dir, train=False, download=True,
                                                     transform=apply_transform)

            # sample training data amongst users
            if args.iid:
                # Sample IID user data from Mnist
                user_groups = mnist_iid(train_dataset, args.num_users)
            else:
                # Sample Non-IID user data from Mnist
                if args.unequal:
                    # Chose unequal splits for every user
                    user_groups = mnist_noniid_unequal(train_dataset, args.num_users)
                else:
                    # Chose equal splits for every user
                    user_groups = mnist_noniid(train_dataset, args.num_users)

        return train_dataset, test_dataset, user_groups
