import numpy as np
import networkx as nx
import random
from mesa.agent import Agent


class AttendanceAgent(Agent):
    def __init__(self,
                 unique_id,
                 model,
                 emotion,
                 expressiveness,
                 susceptbility,
                 amplifier,
                 bias,
                 attend=False) -> None:
        super().__init__(unique_id, model)
        '''
        Args:
            emotion: emotion value of an agent
            openness: parameter for an agent as a sender
            susceptibility: parameter for senders
            amplifier: parameter for senders
            bias: parameter for sender
        '''
        self.emotion = emotion
        self.expressiveness = expressiveness
        self.susceptbility = susceptbility
        self.amplifier = amplifier
        self.bias = bias
        self._next_emotion = None
        self.attend = attend

    def step(self):
        self._next_emotion = self.emotion
        if self.unique_id != self.model.num_agents:
            k1, k2, k3, k4 = self.model.rk4(self.model.dq_dt, self.unique_id,
                                            self.emotion)
            # self.dt has been multiplied in the rk4 function
            self._next_emotion = self.emotion + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    def advance(self):
        self.emotion = self._next_emotion