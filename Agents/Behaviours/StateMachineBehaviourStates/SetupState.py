from spade.behaviour import State

import Config


class SetupState(State):
    """
    State in which the State Machine is initialized.
    """

    async def run(self):
        print("[{}] SETUP".format(self.agent.name))
        self.set_next_state(Config.TRAIN_STATE_AG)
