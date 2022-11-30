import codecs
import datetime
import pickle

from spade.behaviour import CyclicBehaviour
from spade.message import Message
from termcolor import colored

import Config


class ReceiveBehaviour(CyclicBehaviour):
    """
    Behaviour that manages the reception of weights from other agents. These weights are used to apply
    consensus and update the local weights of the agent. The agent then responds with his pre-consensus
    weights.
    """

    async def on_end(self):
        """
        Called when the execution of the State Machine has ended.
        """
        print("[{}] ReceiveBehaviour finished".format(self.agent.name))

    def consensus(self, msg):
        """
        Applies based on weights received from another agent
        :param msg: message containing the weights of the sender agent
        """
        if self.agent.weights is not None and msg.body.split("|")[0] != "None":
            # Process message
            t1 = datetime.datetime.now()
            self.agent.message_history.insert(0,
                                              "{}:{}:{} : Received message from {}".format(str(t1.hour), str(t1.minute),
                                                                            str(t1.second), str(msg.sender).split("/")[0]))
            if str(msg.sender).split("@")[0] in self.agent.message_statistics:
                self.agent.message_statistics[str(msg.sender).split("@")[0]]["receive"] += 1
            else:
                self.agent.message_statistics[str(msg.sender).split("@")[0]] = {"send": 0, "receive": 1}
            weights_and_losses = msg.body.split("|")
            unpickled_neighbour_weights = pickle.loads(codecs.decode(weights_and_losses[0].encode(), "base64"))
            unpickled_neighbour_losses = pickle.loads(codecs.decode(weights_and_losses[1].encode(), "base64"))
            neighbour_max_order = int(weights_and_losses[2])
            if self.agent.max_order < neighbour_max_order:
                self.agent.max_order = neighbour_max_order
                self.agent.epsilon_logger.write_to_file(str(self.agent.max_order))

            unpickled_local_weights = pickle.loads(codecs.decode(self.agent.weights.encode(), "base64"))

            # Apply consensus and update model
            consensus_weights = self.agent.consensus.apply_consensus(unpickled_local_weights,
                                                                     unpickled_neighbour_weights,
                                                                     1 / self.agent.max_order)

            #self.agent.weight_logger.write_to_file(
            #    "CONSENSUS,{},{},{},{}".format(consensus_weights[0]['layer_input.weight'].numpy().flatten()[0],
            #                                   consensus_weights[0]['layer_input.bias'].numpy().flatten()[0],
            #                                   consensus_weights[0]['layer_hidden.weight'].numpy().flatten()[0],
            #                                   consensus_weights[0]['layer_hidden.bias'].numpy().flatten()[0]))

            self.agent.federated_learning.add_new_local_weight_local_losses(consensus_weights[0],
                                                                            unpickled_neighbour_losses)
            self.agent.federated_learning.set_model()

            # Update agent properties
            self.agent.weights = codecs.encode(pickle.dumps(consensus_weights), "base64").decode()
            self.agent.losses = codecs.encode(pickle.dumps(unpickled_neighbour_losses), "base64").decode()
            print(colored("[{}] Applied consensus after receive and updated model".format(self.agent.name), 'red'))

    async def run(self):
        """
        Waits until a message is received, then calls a method to apply consensus based on the weights contained
        in the message.
        """
        # Wait for a message
        msg = await self.receive(timeout=4)
        if msg:
            self.agent.message_logger.write_to_file("RECEIVE,{},{}".format(msg.get_metadata("message_id"), msg.sender))
            now = datetime.datetime.now()
            msg_timestamp = msg.get_metadata("timestamp")
            difference = now - datetime.datetime.strptime(msg_timestamp, "%Y-%m-%d %H:%M:%S.%f")
            difference_seconds = difference.total_seconds()

            if difference_seconds < 3:
                # We need to keep in memory the last message received by the agent
                self.agent.last_message = msg

                response_msg = Message(to=str(self.agent.last_message.sender))
                response_msg.body = str(self.agent.weights).strip() + "|" + str(self.agent.losses).strip() + "|" + str(
                    round(self.agent.max_order, 3))
                response_msg.set_metadata("conversation", "response_data")
                response_msg.set_metadata("timestamp", str(datetime.datetime.now()))
                response_msg.set_metadata("message_id", self.agent.last_message.get_metadata("message_id"))

                # We apply the consensus if the agent is not training
                if self.agent.state_machine_behaviour.current_state != Config.TRAIN_STATE_AG:
                    self.consensus(msg)
                else:
                    self.agent.pending_consensus_messages.append(msg)

                self.agent.message_logger.write_to_file(
                    "SEND_RESPONSE,{},{}".format(response_msg.get_metadata("message_id"),
                                                 self.agent.last_message.sender))
                if str(self.agent.last_message.sender).split("@")[0] in self.agent.message_statistics:
                    self.agent.message_statistics[str(self.agent.last_message.sender).split("@")[0]]["send"] += 1
                else:
                    self.agent.message_statistics[str(self.agent.last_message.sender).split("@")[0]] = {"send": 1,
                                                                                                        "receive": 0}
                t1 = datetime.datetime.now()
                self.agent.message_history.insert(0, "{}:{}:{} : Sent response to {}".format(str(t1.hour),
                                                                                             str(t1.minute),
                                                                                             str(t1.second),
                                                                                             str(self.agent.last_message.sender).split("/")[0]))

                await self.send(response_msg)
            else:
                print("[{}] Received old message".format(self.agent.name))
