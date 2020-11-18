import numpy as np
# The #input and #target nodes, physics equation, each nodes' value range, etc. should together be an environment.


class AbstractScenario(object):
    def __init__(self, num_inputs, num_outputs, interval, trajectory_len, delta, add_noise, g=9.8):
        """[summary]

        Args:
            num_inputs ([type]): [description]
            num_outputs ([type]): [description]
            interval ([type]): [description]
            trajectory_len ([type]): [description]
            delta (bool): [Whether the position output is recoding next step or next step's delta from current step]. Defaults to True.
        """
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.interval = interval
        self.trajectory_len = trajectory_len
        self.delta = delta
        self.add_noise = add_noise
        self.g = g

    def rollout_func(self, permutations):
        raise NotImplementedError


class FrictionSliding(AbstractScenario):
    def __init__(self, num_inputs, num_outputs, interval, trajectory_len, delta, add_noise):
        AbstractScenario.__init__(
            self, num_inputs, num_outputs, interval, trajectory_len, delta, add_noise)

    def rollout_func(self, permutations):
        """
        Euler approximation, one step of sliding down a slope.
        Expect permutations to have order [shapes,colors,mus,thetas,masses,x0s]
        """
        inputs = np.expand_dims(permutations, (2, 3))
        inputs = np.repeat(inputs, self.trajectory_len, axis=2)

        mu = inputs[:, 2, 0, 0]
        theta = inputs[:, 3, 0, 0]

        # 2 outputs: vel and loc
        outputs = np.zeros((inputs.shape[0], self.num_outputs,
                            self.trajectory_len, 1))
        outputs[:, 1, 0, 0] = inputs[:, 5, 0, 0]
        # print(np.nonzero(self.g*(np.sin(theta)-np.cos(theta)*mu) < 0))
        for i in range(1, self.trajectory_len):
            outputs[:, 0, i, 0] = outputs[:, 0, i-1, 0] + \
                self.g*(np.sin(theta)-np.cos(theta)*mu)
            if self.delta:
                outputs[:, 1, i, 0] = outputs[:, 0, i-1, 0]*self.interval+0.5 * \
                    self.g*(np.sin(theta)-np.cos(theta)*mu)*(self.interval**2)
            else:
                outputs[:, 1, i, 0] = outputs[:, 1, i-1, 0] + outputs[:, 0, i-1, 0] * \
                    self.interval+0.5*self.g*(np.sin(theta) -
                                              np.cos(theta)*mu)*(self.interval**2)

        return inputs, outputs


class AirFall(AbstractScenario):
    def __init__(self, num_inputs, num_outputs, interval, trajectory_len, delta, add_noise):
        AbstractScenario.__init__(
            self, num_inputs, num_outputs, interval, trajectory_len, delta, add_noise)

    def rollout_func(self, permutations):
        """
        Euler approximation, one step of free fall with air resistance.
        Expect permutations to have order [rhos/c,shapes,areas,colors,ms,x0s]
        """
        inputs = np.expand_dims(permutations, (2, 3))
        inputs = np.repeat(inputs, self.trajectory_len, axis=2)

        m = inputs[:, 4, 0, 0]
        # rho = inputs[:, 0, 0, 0]
        # area = inputs[:, 2, 0, 0]
        # rho*area/2  # C_d defaults to 1 since it's not measurable
        c = inputs[:, 0, 0, 0]  # In this notation, terminal vel=sqrt(m*g/c)
        # 2 outputs: vel and loc
        outputs = np.zeros((inputs.shape[0], self.num_outputs,
                            self.trajectory_len, 1))
        outputs[:, 1, 0, 0] = inputs[:, 5, 0, 0]

        for i in range(1, self.trajectory_len):
            outputs[:, 0, i, 0] = outputs[:, 0, i-1, 0] + \
                (self.g-c*(outputs[:, 0, i-1, 0]**2)/m)*self.interval
            if self.delta:
                outputs[:, 1, i, 0] = outputs[:, 0, i-1, 0]*self.interval
            else:
                outputs[:, 1, i, 0] = outputs[:, 1, i-1, 0] + \
                    outputs[:, 0, i-1, 0]*self.interval  # +0.5*a*(delta_t**2)
        return inputs, outputs


class FrictionlessSHO(AbstractScenario):
    def __init__(self, num_inputs, num_outputs, interval, trajectory_len, delta, add_noise):
        AbstractScenario.__init__(
            self, num_inputs, num_outputs, interval, trajectory_len, delta, add_noise)

    def rollout_func(self, permutations):
        """
        Euler approximation, one step of Simple Harmonic Oscillator on a frictionless ground, with origin at the wall.
        Expect permutations to have order [shapes, colors, As, ms, ks,spring_lens]
        """
        inputs = np.expand_dims(permutations, (2, 3))
        inputs = np.repeat(inputs, self.trajectory_len, axis=2)

        m = inputs[:, 3, 0, 0]
        k = inputs[:, 4, 0, 0]
        omega = np.sqrt(k/m)
        # print(set(omega))
        # print(2*np.pi/omega)
        outputs = np.zeros((inputs.shape[0], self.num_outputs,
                            self.trajectory_len, 1))
        outputs[:, 1, 0, 0] = inputs[:, 2, 0, 0]

        for i in range(1, self.trajectory_len):
            outputs[:, 0, i, 0] = -outputs[:, 1, i-1, 0]*omega * \
                np.sin(omega*self.interval) + \
                outputs[:, 0, i-1, 0]*np.cos(omega*self.interval)
            outputs[:, 1, i, 0] = outputs[:, 1, i-1, 0] * \
                np.cos(omega*self.interval) + \
                (outputs[:, 0, i-1, 0]/omega)*np.sin(omega*self.interval)

            # outputs[:, 0, i, 0] = outputs[:, 0, i-1, 0] - \
            #     (omega**2)*outputs[:, 1, i-1, 0]*self.interval

            # if self.delta:
            #     outputs[:, 1, i, 0] = outputs[:, 0, i-1, 0]*self.interval
            # else:
            #     outputs[:, 1, i, 0] = outputs[:, 1, i-1, 0] + \
            #         outputs[:, 0, i-1, 0]*self.interval
        # outputs[:, 1, :, 0] += inputs[:, 5, :, 0]

        return inputs, outputs
