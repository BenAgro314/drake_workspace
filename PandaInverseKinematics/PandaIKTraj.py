import numpy as np
from pydrake.all import (
        MultibodyPlant, 
        RollPitchYaw, 
        RotationMatrix,
        AutoDiffXd,
        autoDiffToValueMatrix,
        autoDiffToGradientMatrix,
        JacobianWrtVariable)
from pydrake.solvers.mathematicalprogram import MathematicalProgram, Solve




class PandaIKTraj:
    """ Given a starting gripper pose and a goal gripper pose, find satisfactory
    joint trajectories based on constraints/costs.

    Solves the problem using trajectory optimization 
    """

    # solve conneted IK problems at discrete intervals, and connect them 
    # with a spline (peicewise polynomials)

    def __init__(self, 
            plant, 
            plant_context, 
            model_instance,
            start_pose, # RigidTransform
            goal_pose, # RigidTransform
            num_points,  # the number of intermediate IK solutions
            p_tol = 0.01,
            theta_tol = 0.01,
            avoid_names = [], 
            end_effector = "panda_hand"):
            
        self.plant_f = plant
        self.context_f = plant_context
        self.plant_ad = self.plant_f.ToAutoDiffXd()
        self.context_ad = self.plant_ad.CreateDefaultContext()
        self.context_ad.SetTimeStateAndParametersFrom(self.context_f)
        self.model_instance = model_instance
        self.respect_joint_limits = True

        self.W = self.plant_f.world_frame()
        self.L = self.plant_f.GetFrameByName(end_effector)

        self.num_positions = plant.num_positions(self.model_instance)
        self.num_points = num_points 
        self.start_pose = start_pose
        self.goal_pose = goal_pose
        

        # find joint limits
        joint_indices = plant.GetJointIndices(model_instance)[:self.num_positions] # remove the last joint because they are fixed
        joint_limits = {'lower': [], 'upper': []}
        for ind in joint_indices:
            joint = plant.get_joint(ind)
            joint_limits['lower'].append(joint.position_lower_limits()[0])
            joint_limits['upper'].append(joint.position_upper_limits()[0])

        # identify which positions in the multibody plant are those of the arm
        lower_lims = plant.GetPositionLowerLimits()
        self.lower_ind = np.where(lower_lims == -2.8973)[0][0] 

        # get collision geometries of arm
        self.panda_geom_ids = []
        for i in self.plant_f.GetBodyIndices(self.model_instance):
            b = self.plant_f.get_body(i)
            if (b.name() == "panda_link0"):
                continue 
            self.panda_geom_ids += self.plant_f.GetCollisionGeometriesForBody(b)

        # get collision geometries for the things we don't want to collide with
        self.avoid_geom_ids = []
        for name in avoid_names:
            bodies = self.plant_f.GetBodyIndices(plant.GetModelInstanceByName(name))
            for i in bodies:
                self.avoid_geom_ids += self.plant_f.GetCollisionGeometriesForBody(self.plant_f.get_body(i))

        # create mathematical program and add decision variables
        self.prog = MathematicalProgram()
        self.q = self.prog.NewContinuousVariables(self.num_points + 2, 
                self.num_positions,
                name = "q")

        # add joint limit constraints
        for row in self.q:
            self.prog.AddBoundingBoxConstraint(joint_limits['lower'], joint_limits['upper'], row)

        #add initial and final kinematic constraints 
        self.p_tol = np.ones(3)*p_tol
        self.theta_tol = theta_tol

        ps_WG = self.start_pose.translation()
        pg_WG = self.goal_pose.translation()
        Rs_WG = self.start_pose.rotation()
        Rg_WG = self.goal_pose.rotation()

        self.AddPositionConstraint(ps_WG - self.p_tol, ps_WG + self.p_tol, 0)
        self.AddPositionConstraint(pg_WG - self.p_tol, pg_WG + self.p_tol, -1)
        self.AddOrientationConstraint(Rs_WG, self.theta_tol, 0)
        self.AddOrientationConstraint(Rg_WG, self.theta_tol, -1)

    
        #add constraints on intermediate solutions 
        for i in range(len(self.q) - 1):
            q_now = self.q[i]
            q_next = self.q[i+1]
            self.prog.AddCost((q_next - q_now).dot(q_next - q_now))

    def AddJointCenteringCost(self, q_nominal):
        for row in self.q:
            self.prog.AddQuadraticErrorCost(np.identity(len(row)), q_nominal, row)
        

    def AddPositionConstraint(self, p_WQ_lower, p_WQ_upper, n, p_LQ = np.zeros(3)):
        """ Adds a bounding box constraint on the position of the panda's hand

        p_WQ_lower: the lower coordinate of the bounding box in the world frame (np.array (1,3))
        p_WQ_upper: the upper coordinate of the bounding box in the world frame (np.array (1,3))
        p_LQ: an optional offset from the hand frame to the point Q  (np.array (1,3))
        """
        p_WL = lambda q: self.X_WL(q).translation() + p_LQ 
        self.prog.AddConstraint(p_WL, lb = p_WQ_lower, ub = p_WQ_upper, vars = self.q[n])

    def AddOrientationConstraint(self, R_WD, theta_tol, n):
        diff = lambda q: [self.AngleBetween(q, R_WD)]
        self.prog.AddConstraint(diff, lb = [-theta_tol], ub = [theta_tol], vars = self.q[n])

    def AngleBetween(self, q, R_WD): 
        """ find the relative rotation matrix between the link orientation and the desired orientation"""
        if q.dtype == float:
            plant = self.plant_f
            context = self.context_f
        else:
            plant = self.plant_ad
            context = self.context_ad
            R_WD = R_WD.cast[AutoDiffXd]()
        plant.SetPositions(context, self.model_instance, q)
        R_LW = plant.CalcRelativeTransform(context, self.resolve_frame(plant, self.L), self.resolve_frame(plant, self.W)).rotation()
        R_LD = R_LW.matrix().dot(R_WD.matrix())
        theta = np.arccos(0.5*(R_LD.trace() - 1))
        return abs(theta)

    def X_WL(self, q):
        if q.dtype == float:
            plant = self.plant_f
            context = self.context_f
        else:
            plant = self.plant_ad
            context = self.context_ad
        plant.SetPositions(context, self.model_instance, q)
        return plant.CalcRelativeTransform(context, self.resolve_frame(plant, self.W), self.resolve_frame(plant, self.L))

                
    def get_prog(self):
        return self.prog
    
    def get_q(self):
        return self.q

    @staticmethod
    def resolve_frame(plant, F):
        return plant.GetFrameByName(F.name(), F.model_instance())
