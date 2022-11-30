import copy
import torch
import numpy as np


class Consensus:

    def __init__(self):
        pass

    def apply_consensus(self, own_weights, neighbour_weights, eps):
        """
        Apply the asynchronous consensus between the weights of the agent and those of its neighbour
        :param own_weights: weights of the agent
        :param neighbour_weights: weights of the neighbour
        :param eps: epsilon value
        :return: the new weights post-consensus
        """
        average_weights = copy.deepcopy(own_weights)
        for key in own_weights[0].keys():
            if len(own_weights[0][key]) != len(neighbour_weights[0][key]):
                print("Error - consensus can only be applied to arrays of same length")
                return None

            temp_own = own_weights[0][key].numpy()
            temp_neighbour = neighbour_weights[0][key].numpy()

            temp_own_flat = temp_own.flatten()
            temp_neighbour_flat = temp_neighbour.flatten()

            temp_consensus = temp_own_flat + eps*(temp_neighbour_flat - temp_own_flat)
            average_weights[0][key] = torch.from_numpy(np.reshape(temp_consensus, temp_own.shape))
        return average_weights
