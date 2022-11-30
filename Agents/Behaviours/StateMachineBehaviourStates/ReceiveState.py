import codecs
import datetime
import pickle

from spade.behaviour import State
from termcolor import colored

import Config


class ReceiveState(State):
    """
    State waiting for a response after the agent sends his weights to another agent.
    """

    def __init__(self):
        super().__init__()
        self.conta = 0
        self.list_dict = []

    def consensus(self, msg):
        if self.agent.weights is not None and msg.body.split("|")[0] != "None":
            # Process message
            weights_and_losses = msg.body.split("|")
            unpickled_neighbour_weights = pickle.loads(codecs.decode(weights_and_losses[0].encode(), "base64"))
            unpickled_neighbour_losses = pickle.loads(codecs.decode(weights_and_losses[1].encode(), "base64"))
            neighbour_max_order = int(weights_and_losses[2])
            if self.agent.max_order < neighbour_max_order:
                self.agent.max_order = neighbour_max_order

            unpickled_local_weights = pickle.loads(codecs.decode(self.agent.weights.encode(), "base64"))

            # Apply consensus and update model
            consensus_weights = self.agent.consensus.apply_consensus(unpickled_local_weights,
                                                                     unpickled_neighbour_weights,
                                                                     1 / self.agent.max_order)

            self.agent.federated_learning.add_new_local_weight_local_losses(consensus_weights[0],
                                                                            unpickled_neighbour_losses)
            self.agent.federated_learning.set_model()

            # Update agent properties
            self.agent.weights = codecs.encode(pickle.dumps(consensus_weights), "base64").decode()
            self.agent.losses = codecs.encode(pickle.dumps(unpickled_neighbour_losses), "base64").decode()
            print(colored("[{}] Applied consensus after response and updated model".format(self.agent.name), 'red'))

    async def run(self):
        print("[{}] RECEIVE".format(self.agent.name))
        msg = await self.receive(timeout=10)

        if msg:
            t1 = datetime.datetime.now()
            self.agent.message_history.insert(0, "{}:{}:{} : Received response from {}".format(str(t1.hour),
                                                                                               str(t1.minute),
                                                                                               str(t1.second),
                                                                                               str(msg.sender).split("/")[0]))
            print(colored("[{}] Received response from {}".format(self.agent.name, msg.sender), 'cyan'))
            self.agent.message_logger.write_to_file(
                "RECEIVE_RESPONSE,{},{}".format(msg.get_metadata("message_id"), msg.sender))
            if str(msg.sender).split("@")[0] in self.agent.message_statistics:
                self.agent.message_statistics[str(msg.sender).split("@")[0]]["receive"] += 1
            else:
                self.agent.message_statistics[str(msg.sender).split("@")[0]] = {"send": 0, "receive": 1}

            # Apply consensus and update the local model with new weights
            self.consensus(msg)
            self.set_next_state(Config.TRAIN_STATE_AG)

        else:
            print("[{}] There was an error, no response was received".format(self.agent.name))
            self.set_next_state(Config.TRAIN_STATE_AG)
