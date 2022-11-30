from spade.behaviour import OneShotBehaviour
from termcolor import colored

import Config


class PresenceBehaviour(OneShotBehaviour):
    """
    Manages the presence of the agent, subscribes to its neighbours
    """

    async def on_end(self):
        print("PresenceBehaviour finished")

    def on_available(self, jid, stanza):
        print(colored("[{}] Agent {} is available.".format(self.agent.name, jid.split("@")[0]), 'green'))

    def on_unavailable(self, jid, stanza):
        print(colored("[{}] Agent {} is unavailable.".format(self.agent.name, jid.split("@")[0]), 'green'))
        if jid in self.agent.available_agents:
            self.agent.available_agents.remove(jid)
        self.agent.max_order = len(self.agent.available_agents)
        self.agent.epsilon_logger.write_to_file(str(len(self.agent.available_agents)))

    def on_subscribed(self, jid):
        print(
            colored("[{}] Agent {} has accepted the subscription.".format(self.agent.name, jid.split("@")[0]), 'green'))
        print(colored("[{}] Contacts List: {}".format(self.agent.name, self.agent.presence.get_contacts()), 'green'))

    def on_subscribe(self, jid):
        """
        When an agent asks for subscription, he is added to the available agents list and we subscribe back to him
        :param jid: jid of the agent who is subscribing
        """
        print(colored(
            "[{}] Agent {} asked for subscription. Let's approve it.".format(self.agent.name, jid.split("@")[0]),
            'green'))
        print(colored("[{}] Contacts List: {}".format(self.agent.name, self.agent.presence.get_contacts()), 'green'))

        if jid not in self.agent.available_agents:
            if jid.split('@')[0] in str(self.presence.get_contacts()):
                self.presence.unsubscribe(jid)
            self.agent.available_agents.append(jid)
            self.presence.subscribe(jid)

    async def run(self):
        """
        Attach all the handlers to the Presence behaviour, and subscribe to all neighbours in the graph
        """
        self.presence.on_subscribe = self.on_subscribe
        self.presence.on_subscribed = self.on_subscribed
        self.presence.on_available = self.on_available
        self.presence.on_unavailable = self.on_unavailable
        self.presence.set_available()
        for a in self.agent.neighbours:
            self.presence.subscribe(a + "@" + Config.xmpp_server)
