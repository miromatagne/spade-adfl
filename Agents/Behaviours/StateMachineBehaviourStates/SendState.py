import datetime
import random
import uuid

from spade.behaviour import State
from spade.message import Message

import Config


class SendState(State):
    """
    State in which the agent sends its weights to one of its neighbours.
    """

    def send_message(self, recipient):
        msg = Message(to=recipient)  # Instantiate the message
        # msg.body = "I am sending my weights"
        msg.set_metadata("conversation", "pre_consensus_data")
        msg.set_metadata("timestamp", str(datetime.datetime.now()))

        local_weights = self.agent.weights
        local_losses = self.agent.losses

        if local_weights is None or local_losses is None:
            msg.body = "I don't have any weights yet"
        else:
            msg.body = str(local_weights).strip() + "|" + str(self.agent.losses).strip() + "|" + str(
                round(self.agent.max_order, 3))
            print("Message length : {}".format(len(msg.body)))
        return msg

    async def run(self):
        if len(self.agent.available_agents) > 0:
            print("[{}] SEND".format(self.agent.name))
            receiving_agent_id = random.randint(0, len(self.agent.available_agents) - 1)
            msg = self.send_message(self.agent.available_agents[receiving_agent_id].split("/")[0])
            message_id = str(uuid.uuid4())
            msg.set_metadata("message_id", message_id)
            print("[{}] Sending message to {}".format(self.agent.name,
                                                      self.agent.available_agents[receiving_agent_id].split("/")[0]))
            if self.agent.available_agents[receiving_agent_id].split("@")[0] in self.agent.message_statistics:
                self.agent.message_statistics[self.agent.available_agents[receiving_agent_id].split("@")[0]][
                    "send"] += 1
            else:
                self.agent.message_statistics[self.agent.available_agents[receiving_agent_id].split("@")[0]] = {
                    "send": 1, "receive": 0}
            self.agent.message_logger.write_to_file(
                "SEND,{},{}".format(message_id, self.agent.available_agents[receiving_agent_id].split("/")[0]))
            await self.send(msg)
            t1 = datetime.datetime.now()
            self.agent.message_history.insert(0, "{}:{}:{} : Sent message to {}".format(str(t1.hour), str(t1.minute),
                                                                                        str(t1.second),
                                                                                        self.agent.available_agents[
                                                                                            receiving_agent_id].split(
                                                                                            "/")[0]))
            self.set_next_state(Config.RECEIVE_STATE_AG)
        else:
            self.set_next_state(Config.TRAIN_STATE_AG)