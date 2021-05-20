import numpy as np
from .qspace import QSpace

class Node:

    def __init__(self, q):
        """
        initialize a node with configuration q (np.array)
        """
        self.q = q
        self.parent = None


class RRT:

    def __init__(self, start, goal, is_colliding, q_ranges, max_iter = 1000, goal_sample_rate = 0.05, interp_distances = 0.1):
        """
        initialize a RRT problem

        start: starting configuration np.array
        goal: goal configuration np.array
        collision_checker: a function that intakes a configuration as an np.array and outputs True/False if there is a collision or not
        q_ranges: [(q1_lower, q1_upper), (q2_lower, q2_upper), ... ] is the ranges for each coordinate in configuration space
        max_iter: the maximum number of iterations until we give up
        goal_sample_rate: the probability that we sample the goal configuration
        inter_distances: a np.array or float representing the maximum change that each q can undergo between steps along a linearly interpolated path
        """

        self.qspace = QSpace(q_ranges, interp_distances)

        assert self.qspace.q_valid(start) and self.qspace.q_valid(goal)

        self.start = Node(start)
        self.goal = Node(goal)

        self.is_colliding = is_colliding
        self.max_iter = max_iter
        self.goal_sample_rate = goal_sample_rate
        self.node_list = []
    
    def sample(self):
        if np.random.uniform() > self.goal_sample_rate:
            return Node(self.qspace.sample()) 

        return Node(self.goal.q)

    def nearest_node(self, node):
        distances = [QSpace.distance(node.q, n.q) for n in self.node_list]
        return self.node_list[distances.index(min(distances))]

    def valid_edge(self, n1, n2):
        path = self.qspace.linear_interpolation(n1.q, n2.q)
        for q in path:
            if (self.is_colliding(q)):
                return False
        return True

    def get_plan(self):
        next_node = self.goal.parent
        assert next_node is not None
        plan = [self.goal.q]
        while next_node is not None:
            plan.append(next_node.q)
            next_node = next_node.parent

        plan.reverse()
        return plan
        

    def plan(self):
        self.node_list = [self.start]
        for i in range(self.max_iter):

            # sample for a node
            rnd_node = self.sample()
            # find the nearest node
            nearest_node = self.nearest_node(rnd_node)
            rnd_node.parent = nearest_node
            # check if the edge with the new node is collision free
            if (not self.valid_edge(rnd_node, nearest_node)): 
                self.node_list.append(rnd_node)

            # check if we can go straight to the goal
            if (self.valid_edge(self.node_list[-1], self.goal)):
                self.goal_node.parent = self.node_list[-1]
                return self.get_plan()

        return None
