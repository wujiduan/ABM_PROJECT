import numpy as np
import networkx as nx
from mesa.model import Model
from mesa.time import SimultaneousActivation
from mesa.datacollection import DataCollector
from ABM_agent import AttendanceAgent


def compute_attendance(model):
    attended_num = 0
    for temp_agent in model.schedule.agents:
        if temp_agent.drop_out:
            continue
        attended_num += 1
    return attended_num


class AttendanceModel(Model):
    def __init__(self, num_agents, student_net, max_steps=500) -> None:
        super().__init__()
        self.schedule = SimultaneousActivation(self)
        self.num_agents = num_agents
        self.max_steps = max_steps
        self.attend_step = 50  # Every 50 steps we update the attending decisions of agents
        self.student_net = student_net
        self.dt = 0.2

        # Parameters are set casually
        init_mean_a = 0.
        init_std_a = 1.
        init_mean_v = 0.
        init_std_v = 1.
        tau_min = 0.1
        tau_max = 1

        # The following parameters are taken from the jupyter notebook in the tutorial
        self.gamma_a = 0.9
        self.gamma_v = 0.6
        self.gamma_h = 0.7

        self.d0 = 0.05
        self.d1 = 0.5
        self.d2 = 0.1

        self.b1 = 1
        self.b3 = -1

        self.A_a = 0.3
        self.A_v = 0.3

        self.std_a = 6
        self.std_v = 0.5

        # Initialize all agents with a, v ~ N(*, *), tau ~ Unif[*, *]
        arousals = np.random.normal(init_mean_a, init_std_a, num_agents)
        valences = np.random.normal(init_mean_v, init_std_v, num_agents)
        taus = np.random.uniform(tau_min, tau_max, num_agents)
        thresholds = np.random.normal(1, 1, num_agents)

        # Set configuration for data collection
        self.datacollector = DataCollector(
            agent_reporters={
                "Arousal": lambda agent: agent.a,
                "Valence": lambda agent: agent.v
            },
            model_reporters={"Attendance": compute_attendance})

        for i in range(num_agents):
            temp_agent = AttendanceAgent(i, self, arousals[i], valences[i],
                                         taus[i], thresholds[i])
            self.schedule.add(temp_agent)

        self.datacollector.collect(self)

    def rk4(self, f, y):
        """ Returns k1, k2, k3, k4 according to the Runge-Kutta method
        Args:
            f: euqal to the rate at which y changes
            y: valence, arousal, h+, or h-
        """
        k1 = self.dt * f(y)
        k2 = self.dt * f(y + k1 / 2)
        k3 = self.dt * f(y + k2 / 2)
        k4 = self.dt * f(y + k3)
        return k1, k2, k3, k4

    def step(self):
        '''Advance the model by one step.'''
        self.schedule.step()

        self.datacollector.collect(self)
        if self.schedule.steps > self.max_steps:
            self.running = False
