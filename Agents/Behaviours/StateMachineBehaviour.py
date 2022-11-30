import codecs
import datetime
import pickle
import random
import uuid

from spade.behaviour import State, FSMBehaviour
from spade.message import Message
from termcolor import colored

import Config


class StateMachineBehaviour(FSMBehaviour):
    """
    State Machine that handles the behaviour related to training, sending weights and receiving responses.
    """

    async def on_start(self):
        """
        Called when the State Machine is initialized.
        """
        print("[{}] FSM starting at initial state {}".format(self.agent.name, self.current_state))

    async def on_end(self):
        """
        Called when the execution of the State Machine has ended.
        """
        print("[{}] FSM finished at state {}".format(self.agent.name, self.current_state))
        # await self.agent.stop()