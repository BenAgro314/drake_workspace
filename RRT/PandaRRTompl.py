from pydrake.all import Solve, PiecewisePolynomial, GeometrySet
from ompl import base as ob
from ompl import geometric as og
from PandaInverseKinematics import PandaInverseKinematics
import numpy as np


class PandaRRTompl:

    def __init__(self, plant, scene_graph, 
                 plant_context, scene_graph_context, panda,
                 start_pose, start_time, goal_pose, goal_time, avoid_names,
                 p_tol = 0.01, theta_tol = 0.01,
                 q_nominal = np.array([ 0., 0.55, 0., -1.45, 0., 1.58, 0.])):

        """
        A wrapper class around RRT for motion planning with the franka-panda

        plant: the mulibody plant 
        scene_graph: the scene graph
        plant_context: the plant context
        scene_graph_context: the scene graph context
        panda; the model instance of the panda arm
        start_pose: the desired starting pose of the panda end effector (RigidTransform)
        goal_pose: the  desired ending pose of the panda end effector (RigidTransform)
        p_tol: the tolerance in the translation of the end effector (meters)
        theta_tol: the tolerance in the rotation of the end effector (in radians)
        q_nominal: the "comfortable" joint positions of the panda (np.array)
        """

        self.plant = plant
        self.scene_graph = scene_graph
        self.plant_context = plant_context
        self.scene_graph_context = scene_graph_context
        self.panda = panda
        self.avoid_names = avoid_names
        self.p_tol = p_tol
        self.theta_tol = theta_tol
        self.q_nominal = q_nominal
        self.query_output_port = self.scene_graph.GetOutputPort("query")
        self.num_positions = self.plant.num_positions(self.panda)
        self.start_pose = start_pose
        self.goal_pose = goal_pose

        # construct RRT


        self.fix_panda_collisions()

        # find normal number of collision in starting state
        query_object = self.query_output_port.Eval(self.scene_graph_context)
        collision_pairs = query_object.ComputePointPairPenetration()
        self.num_collide = len(collision_pairs)


        self.avoid_geom_ids = self.get_avoid_geom_ids() 


        self.plan, normalized_costs = self.get_plan()

        # make trajectory from results 
        assert goal_time > start_time
        assert len(self.q_nominal) == self.num_positions
        self.times = (goal_time - start_time)*normalized_costs + start_time


    def get_plan(self):
        """ makes a plan using ompl """
        q_start = self.find_q(self.start_pose)
        q_goal = self.find_q(self.goal_pose)
        joint_limits = self.joint_limits()

        space = ob.RealVectorStateSpace(self.num_positions)
        bounds = ob.RealVectorBounds(self.num_positions)
        for i in range(self.num_positions):
            bounds.setLow(i, joint_limits[i][0])
            bounds.setHigh(i, joint_limits[i][1])
        space.setBounds(bounds)
        si = ob.SpaceInformation(space)
        si.setStateValidityChecker(ob.StateValidityCheckerFn(self.isStateValid))
        start = self.q_to_state(space, q_start)
        goal = self.q_to_state(space, q_goal)
        pdef = ob.ProblemDefinition(si)
        pdef.setStartAndGoalStates(start, goal)
        pdef.setOptimizationObjective(ob.PathLengthOptimizationObjective(si))
        planner = og.RRTstar(si)
        planner.setProblemDefinition(pdef)
        planner.setup()
        solved = planner.solve(20)

        assert solved

        path = pdef.getSolutionPath()
        res = []
        costs = []
        for state in path.getStates():
            q = self.parse_state(state)
            if len(costs) == 0:
                costs.append(0)
            else:
                d = q - res[-1]
                cost = np.sqrt(d.dot(d))
                costs.append(costs[-1] + cost)
            res.append(q)

        return np.array(res), np.array(costs)/costs[-1]
                



    def fix_panda_collisions(self):
        # we remove collisions bewtween adjacent links 
        # note the below indices returns in order with adjacent links
        hand = self.plant.GetModelInstanceByName("hand")
        hand_ids = []
        for i in self.plant.GetBodyIndices(hand):
            b = self.plant.get_body(i)
            hand_ids += self.plant.GetCollisionGeometriesForBody(b)
        hand_set = GeometrySet(hand_ids)
        self.scene_graph.ExcludeCollisionsWithin(self.scene_graph_context, hand_set)

        inds = self.plant.GetBodyIndices(self.panda) 
        for i in range(len(inds)-1):
            body1 = self.plant.get_body(inds[i])
            body2 = self.plant.get_body(inds[i+1])
            geom_set1 = GeometrySet(self.plant.GetCollisionGeometriesForBody(body1))
            self.scene_graph.ExcludeCollisionsWithin(self.scene_graph_context,
                    geom_set1)
            geom_set2 = GeometrySet(self.plant.GetCollisionGeometriesForBody(body2))
            self.scene_graph.ExcludeCollisionsBetween(self.scene_graph_context,
                    geom_set1, geom_set2)

            if i == (len(inds) - 2):
                self.scene_graph.ExcludeCollisionsWithin(self.scene_graph_context,
                        geom_set2)
                self.scene_graph.ExcludeCollisionsBetween(self.scene_graph_context,
                        geom_set1, hand_set)
                self.scene_graph.ExcludeCollisionsBetween(self.scene_graph_context,
                        geom_set2, hand_set)


    def get_avoid_geom_ids(self):

        avoid_geom_ids = []
        for name in self.avoid_names:
            bodies = self.plant.GetBodyIndices(self.plant.GetModelInstanceByName(name))
            for i in bodies:
                avoid_geom_ids += self.plant.GetCollisionGeometriesForBody(self.plant.get_body(i))
        return avoid_geom_ids

    def parse_state(self, state):
        """ parses ompl RealVectorStateSpace::StateType into a numpy array"""
        q = []
        for i in range(self.num_positions):
            q.append(state[i])
        return np.array(q)

    def q_to_state(self, space, q):
        """ turns a python list q into an ompl state"""
        state = ob.State(space)
        for i in range(len(q)):
            state[i] = q[i]
        return state

    def isStateValid(self, state):
        """ wrapper around is_colliding that is passed into ompl"""
        # parse state into q
        q = self.parse_state(state)
        # the state is valid if there are no collisions
        return not self.is_colliding(q)


    def is_colliding(self, q):
        """ returns True if the configuration q results in a collision,
        False otherwise
        """
        # forwards kinematics (setting the position of the panda)
        self.plant.SetPositions(self.plant_context, self.panda, q)
        query_object = self.query_output_port.Eval(self.scene_graph_context)
        collision_pairs = query_object.ComputePointPairPenetration()

        for pair in collision_pairs:
            if pair.id_A in self.avoid_geom_ids or pair.id_B in self.avoid_geom_ids:
                return True

        return False

    def find_q(self, des_pose):
        """ given a desired end effector pose des_pose (RigidTransform),
        returns the joint angles, q, allowing for some specified tolerance
        """
        p_WG = des_pose.translation()
        R_WG = des_pose.rotation()
        p_tol = self.p_tol * np.ones(3)
        ik = PandaInverseKinematics(self.plant, self.plant_context,
                                         self.panda, self.avoid_names)
        ik.AddPositionConstraint(p_WG - p_tol, p_WG + p_tol)
        ik.AddOrientationConstraint(R_WG, self.theta_tol)
        ik.AddMinDistanceConstraint(self.p_tol)
        prog = ik.get_prog()
        q = ik.get_q()

        assert len(q) == self.num_positions

        prog.AddQuadraticErrorCost(np.identity(len(q)), self.q_nominal, q)
        prog.SetInitialGuess(q, self.q_nominal)
        result = Solve(prog)

        assert result.is_success()

        return result.GetSolution(q)

    def joint_limits(self):
        joint_inds = self.plant.GetJointIndices(self.panda)[:self.num_positions]
        joint_limits = []
        for i in joint_inds:
            joint = self.plant.get_joint(i)
            joint_limits.append((joint.position_lower_limits()[0],
                                 joint.position_upper_limits()[0]))
        return joint_limits

    def get_trajectory(self):
        if len(self.plan) < 3: 
            # if the plan is shorter than length 3, we cannot find a cubic
            return PiecewisePolynomial.FirstOrderHold(self.times,
                self.plan[:, 0:self.num_positions].T)
        
        return PiecewisePolynomial.CubicShapePreserving(self.times,
                self.plan[:, 0:self.num_positions].T)



