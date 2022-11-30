from spade.agent import Agent
from spade.template import Template

import Config
from Agents.Behaviours.PresenceBehaviour import PresenceBehaviour
from Agents.Behaviours.ReceiveBehaviour import ReceiveBehaviour
from Agents.Behaviours.StateMachineBehaviour import StateMachineBehaviour
from Agents.Behaviours.StateMachineBehaviourStates.ReceiveState import ReceiveState
from Agents.Behaviours.StateMachineBehaviourStates.SendState import SendState
from Agents.Behaviours.StateMachineBehaviourStates.SetupState import SetupState
from Agents.Behaviours.StateMachineBehaviourStates.TrainState import TrainState
from Consensus.Consensus import Consensus
from Logs.Logger import Logger
from FederatedLearning.Federated import Federated


class FLAgent(Agent):
    def __init__(self, jid: str, password: str, port, dataset, model_type, neighbours, model_path):
        super().__init__(jid, password)
        self.port = port
        self.dataset = dataset
        self.neighbours = neighbours
        self.model_path = model_path

        self.state_machine_behaviour = None
        self.presence_behaviour = None
        self.receive_behaviour = None

        self.last_message = None
        self.available_agents = []

        self.weights = None
        self.losses = []
        self.test_accuracies = []
        self.test_losses = []
        self.train_accuracies = []
        self.train_losses = []

        self.max_order = 2

        self.train_acc = None
        self.train_loss = None
        self.test_acc = None
        self.test_loss = None
        self.message_history = []

        self.pending_consensus_messages = []
        self.message_statistics = {}

        self.consensus = Consensus()

        self.federated_learning = Federated(self.name, self.model_path, self.dataset, model_type)
        self.federated_learning.build_model()
        self.federated_learning.print_model()

        self.weight_logger = Logger("Logs/Weight Logs/" + self.name + ".csv", Config.WEIGHT_LOGGER)
        self.training_logger = Logger("Logs/Training Logs/" + self.name + ".csv", Config.TRAINING_LOGGER)
        self.epsilon_logger = Logger("Logs/Epsilon Logs/" + self.name + ".csv", Config.EPSILON_LOGGER)
        self.message_logger = Logger("Logs/Message Logs/" + self.name + ".csv", Config.MESSAGE_LOGGER)
        self.training_time_logger = Logger("Logs/Training Time Logs/" + self.name + ".csv", Config.TRAINING_TIME_LOGGER)

    def list_to_str(self, msg_hist):
        """
        Creates the message history string to be displayed in the agent interface
        :param msg_hist: list of history of messages
        :return:
        """
        res = ""
        for i in msg_hist:
            res += i
            res += "\n"
        return res

    def agent_web_controller(self, request):
        """
        Sends data to the agent's web interface
        """
        history_str = self.list_to_str(self.message_history)
        epoch_range = list(range(1, len(self.test_accuracies) + 1))
        self.available_agents.sort()
        active_neighbours_recv = []
        active_neighbours_send = []

        for i in self.available_agents:
            name = i.split("@")[0]
            if name in self.message_statistics:
                active_neighbours_recv.append(self.message_statistics[name]["receive"])
                active_neighbours_send.append(self.message_statistics[name]["send"])

        if len(self.test_accuracies) == 0:
            epoch_range = []

        print(self.available_agents)
        str_available_agents = []

        for i in self.available_agents:
            str_available_agents.append(str(i))
        
        return {"last_message": self.last_message, "train_acc": self.train_acc, "train_loss": self.train_loss,
                "test_acc": self.test_acc, "test_loss": self.test_loss, "message_history": history_str,
                "epochs": epoch_range, "test_accuracies": self.test_accuracies, "test_losses": self.test_losses,
                "train_accuracies": self.train_accuracies, "train_losses": self.train_losses,
                "received_message_statistics": active_neighbours_recv,
                "sent_message_statistics": active_neighbours_send, "available_agents": str_available_agents,
                "nb_available_agents": len(str_available_agents)}

    async def agent_post_receiver(self, request):
        form = await request.post()

    async def stop_agent(self, request):
        self.state_machine_behaviour.kill()
        self.presence_behaviour.kill()
        self.receive_behaviour.kill()
        await self.stop()

    async def setup(self):
        print("[{}] Agent is running".format(self.name))
        self.web.add_get("/agent", self.agent_web_controller, "Agents/Interfaces/agent.html")
        self.web.add_post("/submit", self.agent_post_receiver, None)
        self.web.add_get("/agent/stop", self.stop_agent, None)
        self.web.start(port=self.port)

        self.state_machine_behaviour = StateMachineBehaviour()
        self.state_machine_behaviour.add_state(name=Config.SETUP_STATE_AG, state=SetupState(), initial=True)
        self.state_machine_behaviour.add_state(name=Config.RECEIVE_STATE_AG, state=ReceiveState())
        self.state_machine_behaviour.add_state(name=Config.TRAIN_STATE_AG, state=TrainState())
        self.state_machine_behaviour.add_state(name=Config.SEND_STATE_AG, state=SendState())
        self.state_machine_behaviour.add_transition(source=Config.SETUP_STATE_AG, dest=Config.TRAIN_STATE_AG)
        self.state_machine_behaviour.add_transition(source=Config.TRAIN_STATE_AG, dest=Config.SEND_STATE_AG)
        self.state_machine_behaviour.add_transition(source=Config.SEND_STATE_AG, dest=Config.RECEIVE_STATE_AG)
        self.state_machine_behaviour.add_transition(source=Config.RECEIVE_STATE_AG, dest=Config.TRAIN_STATE_AG)
        self.state_machine_behaviour.add_transition(source=Config.SEND_STATE_AG, dest=Config.TRAIN_STATE_AG)

        state_machine_template = Template()
        state_machine_template.set_metadata("conversation", "response_data")

        self.add_behaviour(self.state_machine_behaviour, state_machine_template)

        for a in self.presence.get_contacts():
            self.presence.unsubscribe(str(a))

        self.presence_behaviour = PresenceBehaviour()
        self.add_behaviour(self.presence_behaviour)

        receive_template = Template()
        receive_template.set_metadata("conversation", "pre_consensus_data")
        self.receive_behaviour = ReceiveBehaviour()
        self.add_behaviour(self.receive_behaviour, receive_template)
