import random
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
    def __init__(self,
                 num_agents,
                 adjacencyMatrix,
                 seed,
                 openness_teacher,
                 max_steps=500,
                 lecture_duration=50,
                 dt=0.1) -> None:
        super().__init__()

        random.seed(seed)
        self.schedule = SimultaneousActivation(self)
        self.num_agents = num_agents
        self.max_steps = max_steps
        self.adjacency_matrix = adjacencyMatrix
        self.lecture_duration = lecture_duration
        self.openness_teacher = openness_teacher
        self.dt = dt
        self.attended_list = []
        self.interaction_network = None

        # Set configuration for data collection
        self.datacollector = DataCollector(
            agent_reporters={"Emotion": lambda agent: agent.emotion},
            model_reporters={"Attendance": "attendance_rate"})

        # Initial emotions obey the normal distribution
        initial_emotion_mu = 0.6
        initial_emotion_sigma = 2
        initial_emotions = np.random.normal(initial_emotion_mu,
                                            initial_emotion_sigma, num_agents)
        initial_emotions = [max(0, min(1, x)) for x in initial_emotions]
        expressivenesses = np.random_sample(size=num_agents)
        susceptbilities = np.random_sample(size=num_agents)
        amplifiers = np.random_sample(size=num_agents)
        biases = np.random_sample(size=num_agents)

        for i in range(num_agents):
            temp_agent = AttendanceAgent(i, self, initial_emotions[i],
                                         expressivenesses[i],
                                         susceptbilities[i], amplifiers[i],
                                         biases[i])
            self.schedule.add(temp_agent)

        self.UpdateAttendedList()
        self.datacollector.collect(self)

    @property
    def interaction_num(self):
        return self.num_agents * 2

    @property
    def attendance_rate(self):
        return len(self.attended_list) / self.num_agents

    def UpdateAttendedList(self):
        self.attended_list.clear()
        for i in range(self.num_agents):
            temp_agent = self.schedule.agents[i]
            rn = np.random.rand()
            if rn < temp_agent.emotion:
                self.attended_list.append(i)

    def rk4(self, f, i, y):
        """ Returns k1, k2, k3, k4 according to the Runge-Kutta method
        Args:
            f: euqal to the rate at which y changes
            y: 
        """
        k1 = self.dt * f(i)
        k2 = self.dt * f(i, y + k1 / 2)
        k3 = self.dt * f(i, y + k2 / 2)
        k4 = self.dt * f(i, y + k3)
        return k1, k2, k3, k4

    def step(self):
        '''Advance the model by one step.'''
        if self.schedule.steps % self.lecture_duration == 0:
            self.UpdateAttendedList()
        # the interaction_network should always have num_agents nodes no matter
        # how the attended list evolves
        self.interaction_network = self.GenerateInteractions(
            self.attended_list, self.adjacency_matrix, self.interaction_num)
        # the following computations are for the emotion updates
        interaction_crf_mat = nx.adjacency_matrix(self.interaction_network)
        self.w_denominators = np.zeros(self.num_agents)
        self.q_stars = np.zeros(self.num_agents)
        self.gammas = np.zeros(self.num_agents)

        for temp_sender in self.attended_list:
            temp_receivers = interaction_crf_mat[temp_sender].indices
            temp_alphas = interaction_crf_mat[temp_sender].data
            for i in range(len(temp_receivers)):
                r = temp_receivers[i]
                self.w_denominators[r] += self.schedule.agents[
                    r].expressiveness * temp_alphas[i]

        for temp_sender in self.attended_list:
            temp_receivers = interaction_crf_mat[temp_sender].indices
            temp_alphas = interaction_crf_mat[temp_sender].data
            for i in range(len(temp_receivers)):
                r = temp_receivers[i]
                self.q_stars[r] += self.schedule.agents[
                    temp_sender].expressiveness * temp_alphas[
                        i] / self.w_denominators[r] * self.schedule.agents[
                            temp_sender].emotion
                self.gammas[r] += self.schedule.agents[
                    temp_sender].expressiveness * temp_alphas[
                        i] * self.schedule.agents[r].susceptbility

        self.schedule.step()
        self.datacollector.collect(self)
        if self.schedule.steps > self.max_steps:
            self.running = False

    def dq_dt(self, id, emotion):
        temp_agent = self.schedule.agents[id]
        eta = temp_agent.amplifier
        beta = temp_agent.bias

        PI = 1 - (1 - self.q_stars[id]) * (1 - emotion)
        NI = self.q_stars[id] * emotion
        return self.gammas[id] * (eta * (beta * PI + (1 - beta) * NI) +
                                  (1 - eta) * self.q_stars[id] - emotion)

    def GenerateInteractions(attendedList, adjacencyMatrix, S):
        pass
