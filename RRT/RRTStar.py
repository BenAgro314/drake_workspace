import numpy as np
from .RRT import RRT
from .qspace import QSpace

class RRTStar(RRT):

    def __init__(self, start, goal, is_colliding, collide_checker,
                 q_ranges, max_iter = 1000, goal_sample_rate = 0.05, 
                 interp_distances = 0.1, beta = 15):
        """
        initialize a RRT* problem

        start: starting configuration np.array
        goal: goal configuration np.array
        collision_checker: a function that intakes a configuration as an np.array and outputs True/False if there is a collision or not
        collide_checker: the instance of the calss that runs is_colliding
        q_ranges: [(q1_lower, q1_upper), (q2_lower, q2_upper), ... ] is the ranges for each coordinate in configuration space
        max_iter: the maximum number of iterations until we give up
        goal_sample_rate: the probability that we sample the goal configuration
        inter_distances: a np.array or float representing the maximum change that each q can undergo between steps along a linearly interpolated path
        beta: the parameter for selecting the nearest nodes (see: https://journals.sagepub.com/doi/10.5772/56718)
        """

        super().__init__(start, goal, is_colliding, collide_checker,
                         q_ranges, max_iter, goal_sample_rate, interp_distances)
        self.beta = beta 

    def near_nodes_inds(self, node):
        """ return the indices of the nearest nodes to node (in configuratino space)"""
        distances = [QSpace.distance(node.q, n.q) for n in self.node_list]
        # TODO(ben): figure out why we use this heuristic
        n = len(self.node_list) + 1.0
        r = self.beta*np.sqrt(np.log(n)/n)
        near_inds = [distances.index(d) for d in distances if d < r]
        return near_inds

    def new_cost(self, from_node, to_node):
        """returns to_node's new cost if from_node was it's parent"""
        return from_node.cost + QSpace.distance(from_node.q, to_node.q)

    def choose_parent(self, new_node, near_inds):
        """ set new_node.parent to the lowest resulting cost parent 
        in self.node_list at near_inds, updates cost of new_node"""

        min_cost = np.inf
        best_near_node = None

        for ind in near_inds:
            near_node = self.node_list[ind]
            if self.valid_edge(new_node, near_node):
                cost = self.new_cost(near_node, new_node)
                if cost < min_cost:
                    best_near_node = near_node
                    min_cost = cost

        #print("parent cost:", min_cost)
        new_node.cost = min_cost
        new_node.parent = best_near_node

    def propogate_cost_to_leaves(self, parent_node):
        """ updates the cost of all the leaves of parent_node"""
        # TODO(ben): add children attribute to nodes so this is more efficient
        for node in self.node_list:
            if node.parent == parent_node:
                node.cost = self.new_cost(parent_node, node)
                self.propogate_cost_to_leaves(node)

    def rewire(self, new_node, near_inds):
        """ re-set the parents of the nodes in given by near_inds to new_node if it
        will result in them having a lower cost (rewiring)"""

        for ind in near_inds:
            near_node = self.node_list[ind]
            if self.valid_edge(new_node, near_node):
                cost = self.new_cost(new_node, near_node)
                if cost < near_node.cost:
                    near_node.parent = new_node

        self.propogate_cost_to_leaves(new_node)

    def best_goal_node_index(self):
        """find the lowest cost node to goal """
        min_cost = np.inf
        best_goal_node_ind = None
        for i in range(len(self.node_list)):
            node = self.node_list[i]
            if self.valid_edge(self.goal, node):
                #print("valid:", i)
                cost = node.cost + QSpace.distance(node.q, self.goal.q)
                if cost < min_cost:
                    min_cost = cost
                    best_goal_node_ind = i
        return best_goal_node_ind, min_cost

    def plan(self):
        """ Find the motion plan in configuration space via RRT*

        Returns the plan (list of configuration vectors) if it exists, along
        with the normalized aggregate cost at each step in that plan

        Returns None,None if the plan could not be found
        """
        self.node_list = [self.start]

        for i in range(self.max_iter):
            rnd_node = self.sample()
            nearest_node, new_cost = self.nearest_node(rnd_node)
            #print(self.node_list.index(nearest_node))

            if (self.valid_edge(rnd_node, nearest_node)): 
                near_inds = self.near_nodes_inds(rnd_node)
                self.choose_parent(rnd_node, near_inds)
                self.node_list.append(rnd_node)
                self.rewire(rnd_node, near_inds)

     

        last_ind, min_cost = self.best_goal_node_index()

        if last_ind is not None:
            self.goal.parent = self.node_list[last_ind]
            self.goal.cost = min_cost
            return self.get_plan()

        return None, None

