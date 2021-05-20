import numpy as np

class QSpace: # the configuation space of the robot

    def __init__(self, q_ranges, max_distances):
        '''
        Defines the confiuration space for a robot

        q_ranges = [(q1_lower, q1_upper), (q2_lower, q2_upper), ... ] is the ranges for each coordinate in configuration space

        max_distances: a np.array or float of the maximum change that each q can undergo between steps along a linear interpolation
        '''
        self.q_ranges = q_ranges

        for r in q_ranges:
            assert r[0] <= r[1]

        self.max_distances = max_distances
        if (type(max_distances) == float or type(max_distances) == int):
            self.max_distances = [max_distances]*len(self.q_ranges)

        assert len(self.q_ranges) == len(self.max_distances)

    @staticmethod
    def distance(q1, q2):
        '''
        returns the distance (float) in configuration space given by q1 and q2 

        q1, q2: numpy arrays represting points in configuration space
        '''
        return np.sqrt((q2-q1).dot(q2-q1))

    def sample(self):
        '''
        returns a random sample point (np.array) in the configuration space
        '''
        return np.array([ ((high - low)*np.random.uniform() + low) for low,high in self.q_ranges ])

    def q_valid(self, q):
        ''' 
        returns true iff q is a valid configuration
            - it must have the correct length
            - it must be in the ranges

        q: configuration np.array 
        '''
        if len(q) != len(self.q_ranges):
            return False
        for i in range(len(q)):
            r = self.q_ranges[i]
            if (r[0] > q[i]) or (r[1] < q[i]):
                return False
        return True

    def linear_interpolation(self, q1, q2):
        '''
        returns a np.array of configurations connecting q1 (np.array) and q2 (np.array), 
        with each confiuration being seperated by at most max_distance

        '''
        assert self.q_valid(q1) and self.q_valid(q2)


        diff = q2 - q1
        samples = max([int(np.ceil(abs(d)/max_d)) for d,max_d in zip(diff, self.max_distances)])
        samples = max(2, samples) # get at least two samples (start and end)
        step = diff/(samples - 1) # how far to step each coordinate

        return np.array([q1 + step*i for i in range(samples)])


        

