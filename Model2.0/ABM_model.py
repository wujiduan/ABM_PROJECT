import random
import copy
import math
import numpy as np
import networkx as nx
from mesa.model import Model
from mesa.time import SimultaneousActivation
from mesa.datacollection import DataCollector
from ABM_agent import AttendanceAgent


class AttendanceModel(Model):
    def __init__(self,
                 num_agents,
                 adjacencyMatrix,
                 seed,
                 expressiveness_teacher,
                 initial_emotion_teacher,
                 updateAdj,
                 friendship_increase,
                 initial_emotion_mu,
                 initial_emotion_sigma,
                 my_lambda,
                 teacher_send_alpha_sigma,
                 divided_group,
                 group_num,
                 alpha_t_lowerbound,
                 max_steps=500,
                 lecture_duration=50,
                 dt=0.1, interactions_multiplier=2) -> None:
        super().__init__()

        random.seed(seed)
        self.schedule = SimultaneousActivation(self)
        self.num_agents = num_agents
        self.max_steps = max_steps
        self.adjacency_matrix = adjacencyMatrix
        self.lecture_duration = lecture_duration
        self.expressiveness_teacher = expressiveness_teacher
        self.initial_emotion_teacher = initial_emotion_teacher
        self.dt = dt
        self.updateAdjMat = updateAdj
        self.friendship_increase = friendship_increase
        self.my_lambda = my_lambda
        self.teacher_send_alpha_sigma = teacher_send_alpha_sigma
        self.divided_group = divided_group
        self.alpha_t_lowerbound = alpha_t_lowerbound
        self.group_num = 1
        if self.divided_group:
            self.group_num = group_num
        self.attended_list = []
        self.interaction_network = None
        self.senders = []
        self.receivers = []
        self.interactions_multiplier=interactions_multiplier

        # Set configuration for data collection
        self.datacollector = DataCollector(
            agent_reporters={"Emotion": lambda agent: agent.emotion,
                            "Attend": lambda agent: agent.attend},
                                
            model_reporters={
                "Attendance": "attendance_rate",
                "adjacencyMatrix": "adjacency_matrix",
                "attendedList" : "attended_list_updated",
                
            })

        # Initial emotions obey the normal distribution
        initial_emotions = np.clip(
            np.random.normal(initial_emotion_mu, initial_emotion_sigma,
                             num_agents), 0, 1)
        expressivenesses = np.random.random_sample(size=num_agents)
        susceptbilities = np.random.random_sample(size=num_agents)
        amplifiers = np.random.random_sample(size=num_agents)
        biases = np.random.random_sample(size=num_agents)

        for i in range(num_agents):
            temp_agent = AttendanceAgent(i, self, initial_emotions[i],
                                         expressivenesses[i],
                                         susceptbilities[i], amplifiers[i],
                                         biases[i])
            self.schedule.add(temp_agent)
            self.attended_list.append(i)
        '''
        Add the teacher node, since the teacher only act as a sender, the expressiveness and the initial emotion
        are the only parameters we need to consider, the susceptibility of the teacher is set to 0, which ensures
        the emotion of the teacher is a constant
        If we set the expressiveness of the teacher to 0, which means we don't consider teacher node in our case
        '''
        teacher = AttendanceAgent(self.num_agents, self,
                                  initial_emotion_teacher,
                                  expressiveness_teacher, 0, 0, 0)
        self.schedule.add(teacher)

    @property
    def interaction_num(self):
        return int(self.num_agents * self.interactions_multiplier)

    @property
    def attendance_rate(self):
        return len(self.attended_list) / self.num_agents
    
    @property   
    def attended_list_updated(self): 
        return self.attended_list
 
    
    def UpdateAttendedList(self):
        k_drop=7
        k_rec=15
        # the bigger the k the lesser likely is a drop
        for i in range(self.num_agents):
            temp_agent = self.schedule.agents[i]
            
            if i in self.attended_list:
                rn = np.random.rand()**k_drop
                
                if rn > temp_agent.emotion:
                    self.attended_list.remove(i)
                    temp_agent.attend=False
            else: 
                rn = np.random.rand()**k_rec
                
                if  rn > temp_agent.emotion:
                    self.attended_list.append(i)
                    temp_agent.attend=True
                      
            
                
    

    def rk4(self, f, i, y):
        """ Returns k1, k2, k3, k4 according to the Runge-Kutta method
        Args:
            f: euqal to the rate at which y changes
            y: dq_dt
        """
        k1 = self.dt * f(i, y)
        k2 = self.dt * f(i, y + k1 / 2)
        k3 = self.dt * f(i, y + k2 / 2)
        k4 = self.dt * f(i, y + k3)
        return k1, k2, k3, k4

    def step(self):
        '''Advance the model by one step.'''
        if self.schedule.steps % self.lecture_duration == 0:
            self.UpdateAttendedList()
        ''' 
        The interaction_network should always have num_agents nodes no matter 
        how the attended list evolves
        '''
        self.GenerateInteractions()

        # the following computations are for the emotion updates
        interaction_crf_mat = nx.adjacency_matrix(self.interaction_network)
        self.w_denominators = np.zeros(self.num_agents)
        self.q_stars = np.zeros(self.num_agents)
        self.gammas = np.zeros(self.num_agents)
        sender_list = self.senders

        for temp_sender in sender_list:
            temp_receivers = interaction_crf_mat[temp_sender].indices
            temp_alphas = interaction_crf_mat[temp_sender].data
            for i in range(len(temp_receivers)):
                r = temp_receivers[i]
                self.w_denominators[r] += self.schedule.agents[
                    temp_sender].expressiveness * temp_alphas[i]

        for r in self.receivers:
            if self.w_denominators[r] <= 0:
                print("denominators become zero!")
                print(self.senders)
                print("receivers:")
                print(self.receivers)
                print("attendedList:")
                print(self.attended_list)
                print("steps:", self.schedule.steps)
                quit()

        for temp_sender in sender_list:
            temp_receivers = interaction_crf_mat[temp_sender].indices
            temp_alphas = interaction_crf_mat[temp_sender].data
            for i in range(len(temp_receivers)):
                r = temp_receivers[i]
                if self.w_denominators[r] <= 0:
                    print("denominators become zero!")
                    print(self.senders)
                    print("receivers:")
                    print(self.receivers)
                    print("attendedList:")
                    print(self.attended_list)
                    print("steps:", self.schedule.steps)
                    print("temp sender:", temp_sender)
                    print("temp receiver:", r)
                    print("temp alphas:", temp_alphas)
                    quit()
                self.q_stars[r] += self.schedule.agents[
                    temp_sender].expressiveness * temp_alphas[
                        i] / self.w_denominators[r] * self.schedule.agents[
                            temp_sender].emotion
                self.gammas[r] += self.schedule.agents[
                    temp_sender].expressiveness * temp_alphas[
                        i] * self.schedule.agents[r].susceptbility

        self.schedule.step()
        self.datacollector.collect(self)
        if self.schedule.steps >= self.max_steps:
            self.running = False

    def dq_dt(self, id, emotion):
        temp_agent = self.schedule.agents[id]
        eta = temp_agent.amplifier
        beta = temp_agent.bias

        PI = 1 - (1 - self.q_stars[id]) * (1 - emotion)
        NI = self.q_stars[id] * emotion
        return self.gammas[id] * (eta * (beta * PI + (1 - beta) * NI) +
                                  (1 - eta) * self.q_stars[id] - emotion)

    def GenerateInteractions(self):

        attended_num = len(self.attended_list)
        # create a new empty graph, the nodes are fixed
        self.interaction_network = nx.empty_graph(self.num_agents + 1,
                                                  create_using=nx.DiGraph)
        self.senders = []
        self.receivers = []
        if attended_num > 0:
            temp_attended_list = self.attended_list.copy()
            random.shuffle(temp_attended_list)
            interactions = [1 for _ in range(self.interaction_num)]
            subgroups = np.array_split(temp_attended_list, self.group_num)
            sub_interaction_nums = np.array_split(interactions, self.group_num)
            for i in range(self.group_num):
                temp_subgroup = subgroups[i]
                temp_interaction_num = sub_interaction_nums[i].sum()
                self.GenerateSubgroupInteractions(temp_subgroup,
                                                  temp_interaction_num)
            # only update the students adjacency matrix
            if self.updateAdjMat:
                self.UpdateAdjacencyMatrix()

            # add the interactions between the teacher node and the student nodes
            # teacher_node.unique_id = self.num_agents
            self.senders.append(self.num_agents)
            alphas_t = np.clip(
                np.random.normal(self.expressiveness_teacher,
                                 self.teacher_send_alpha_sigma, attended_num),
                self.alpha_t_lowerbound, 1)
            for i in range(attended_num):
                self.interaction_network.add_edge(self.num_agents,
                                                  self.attended_list[i],
                                                  weight=alphas_t[i])

    # generate interactions within each group
    def GenerateSubgroupInteractions(self, group, group_interaction_num):

        # the 1-person groups are ignored in our model
        group = list(group)
        if len(group) > 1:
            group_senders = []
            group_receivers = []
            probabilities_sender = []
            for temp_sender in group:
                probabilities_sender.append(
                    self.schedule.agents[temp_sender].emotion)
            group_senders = random.choices(group,
                                           weights=probabilities_sender,
                                           k=group_interaction_num)
            for temp_sender in group_senders:
                probabilities_receiver = np.zeros(self.num_agents)
                probabilities_receiver[group] = self.adjacency_matrix[
                    temp_sender, :][group] + 1
                probabilities_receiver[temp_sender] = 0
                group_receivers.append(
                    random.choices(range(self.num_agents),
                                   weights=probabilities_receiver,
                                   k=1)[0])
            alphas = np.random.rand(group_interaction_num)
            #add the influence of previous friendship
            # - to bound the alphas within the [0, 1] interval, I suggest using multiplications instead of additions
            ## addition
            # for i in range(S):
            #     alphas[i] = (alphas[i] +
            #                  adjacencyMatrix[senders[i]][receivers[i]]) / 2
            ## multiplication
            for i in range(group_interaction_num):
                alphas[i] *= self.adjacency_matrix[group_senders[i]][
                    group_receivers[i]]
            for i in range(group_interaction_num):
                self.interaction_network.add_edge(group_senders[i],
                                                  group_receivers[i],
                                                  weight=alphas[i])

            self.senders += group_senders
            self.receivers += group_receivers

    def UpdateAdjacencyMatrix(self):
        # I suggest a smaller number for friendship_increase value such as 0.01
        # or new_friendship = old_friendship * lambda + 1 * (1 - lambda)
        # new - old = (1 - lambda) * (1 - old)
        # - this can save us efforts to confine values in [0, 1]
        # - also, "the higher the old_friendship is, the less the absolute increase will be"
        # friendship_increase = self.friendship_increase
        # for s in senders:
        #     self.adjacency_matrix[s][receivers] += friendship_increase
        # for r in receivers:
        #     self.adjacency_matrix[r][senders] += friendship_increase
        my_lambda = self.my_lambda
        senders = self.senders
        receivers = self.receivers
        for s in senders:
            self.adjacency_matrix[s][receivers] = self.adjacency_matrix[s][
                receivers] * my_lambda + 1 - my_lambda
        for r in receivers:
            self.adjacency_matrix[r][senders] = self.adjacency_matrix[r][
                senders] * my_lambda + 1 - my_lambda

        self.adjacency_matrix = np.clip(self.adjacency_matrix, 0, 1)