import numpy as np
import random
from mesa.agent import Agent


class AttendanceAgent(Agent):
    def __init__(self, unique_id, model, arousal, valence, tau,
                 threshold) -> None:
        super().__init__(unique_id, model)
        '''
        Args:
            valence: pleasure related to an emotion
            arousal: personal activity induced by the emotion
            tau: threshold of the arousal for an agent to express its valence (happy or sad)

        Arousal triggers, but valence sets the magnitude & the sign of the expression
        '''
        self.v = valence
        self.a = arousal
        self.tau = tau
        self.fallen_out = False
        self.drop_out = False
        self._next_v = None
        self._next_a = None
        self.attend_threshold = threshold
        self.h_pos = 0
        self.h_neg = 0

    @property
    def s(self):
        # s is 0 when the agent does not express its valence,
        # otherwise it is equal to the sign of its valence
        return 0 if self.a < self.tau else np.sign(self.v)

    @property
    def N_pos(self):
        res = 0
        nn_ids = list(self.model.student_net.predecessors[self.unique_id])
        for nn_id in nn_ids:
            nn = self.model.schedule.agents[nn_id]
            if nn.v > 0:
                res += 1
        return res

    @property
    def N_neg(self):
        pass

    @property
    def W_sum(self):
        w_pos_sum = 0.
        w_neg_sum = 0.
        nns = [
            (u, d['weight']) if u != self.unique_id else (v, d['weight'])
            for (u, v,
                 d) in self.model.student_net.edges(self.unique_id, data=True)
        ]
        for nn_id, w in nns:
            nn = self.model.schedule.agents[nn_id]
            if nn.v > 0:
                w_pos_sum += w
            else:
                w_neg_sum -= w
        return [w_pos_sum, w_neg_sum]

    def dh_pos_dt(self, h_pos):
        # ODE for modelling the dynamics of communication field h+ (see slide 12)
        # return -self.gamma_h * h_pos + self.s * self.N_pos
        return -self.model.gamma_h * h_pos + self.s * self.W_sum[
            0]  # positive weights

    def dh_neg_dt(self, h_neg):
        # ODE for modelling the dynamics of communication field h- (see slide 12)
        # return -self.gamma_h * h_neg + self.s * self.N_neg
        return -self.model.gamma_h * h_neg + self.s * self.W_sum[
            1]  # negative weights
        '''
        Original version in [1]
        '''
        # if v >= 0:
        #     return -gamma_v * v + h_pos * (b1 * v +
        #                                    b3 * pow(v, 3)) + A_v * xi
        # else:
        #     return -gamma_v * v + h_neg * (b1 * v +
        #                                    b3 * pow(v, 3)) + A_v * xi

    def da_dt(self, a):
        # ODE for modelling the dynamics of subthreshold arousal (see slide 28)
        gamma_a = self.model.gamma_a
        d0 = self.model.d0
        d1 = self.model.d1
        d2 = self.model.d2
        h_pos = self.h_pos
        h_neg = self.h_neg
        h = h_pos + h_neg
        A_a = self.model.A_a
        xi = np.random.normal(0, self.model.std_a)
        return -gamma_a * a + h * (d0 + d1 * a + d2 * a * a) + A_a * xi

    def dv_dt(self, v):
        # ODE for modelling the dynamics of valence (see slide 28)
        # set b = 0, b_0 = 0, b_2 = 0
        gamma_v = self.model.gamma_v
        h_pos = self.h_pos
        h_neg = self.h_neg
        b1 = self.model.b1
        b3 = self.model.b3
        A_v = self.model.A_v
        xi = np.random.normal(0, self.model.std_v)
        return -gamma_v * v + (h_pos - h_neg) * (b1 * v +
                                                 b3 * pow(v, 3)) + A_v * xi

    def step(self):
        k1, k2, k3, k4 = self.model.rk4(self.dv_dt, self.v)
        self._next_v = self.v + (k1 + 2 * k2 + 2 * k3 + k4) / 6

        # Update communication field h+
        k1, k2, k3, k4 = self.model.rk4(self.dh_pos_dt, self.h_pos)
        delta = (k1 + 2 * k2 + 2 * k3 + k4) / 6
        self.h_pos = self.h_pos + delta
        # Update communication field h-
        k1_n, k2_n, k3_n, k4_n = self.model.rk4(self.dh_neg_dt, self.h_neg)
        delta_neg = (k1_n + 2 * k2_n + 2 * k3_n + k4_n) / 6
        self.h_neg = self.h_neg + delta_neg

        if not self.fallen_out:
            if self.s == 0:
                k1_a, k2_a, k3_a, k4_a = self.model.rk4(self.da_dt, self.a)
                self._next_a = self.a + (k1_a + 2 * k2_a + 2 * k3_a + k4_a) / 6
            else:
                self._next_a = 0
        else:
            self._next_a = 0
            self.fallen_out = False

    def advance(self):
        self.v = self._next_v
        self.a = self._next_a
        if self.a < -10:
            self.fallen_out = True
            self.a = 0

        # For a larger time step, if the valence is low enough (below the threshold)
        # the student then chooses to skip the course for once
        if self.model.schedule.steps % self.model.attend_step == 0:
            if self.v < self.attend_threshold:
                self.drop_out = True
            else:
                self.drop_out = False