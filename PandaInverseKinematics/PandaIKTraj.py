import numpy as np
from pydrake.all import (
        MultibodyPlant, 
        RollPitchYaw, 
        RotationMatrix,
        AutoDiffXd,
        autoDiffToValueMatrix,
        autoDiffToGradientMatrix,
        JacobianWrtVariable,
        PiecewisePolynomial)
from pydrake.solvers.mathematicalprogram import MathematicalProgram, Solve


class Waypoint:

    def __init__(self, pose, time):
        self.pose = pose
        self.time = time

class Trajectory:

    def __init__(self, start_pose):
        self.waypoints = [Waypoint(start_pose, 0)]
        self.divisions = []

    def add_waypoint(self, pose, time, N):
        self.waypoints.append(Waypoint(pose, time))
        self.divisions.append(N)

    def length(self):
        return len(self.waypoints)


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
            trajectory,
            p_tol = 0.01,
            theta_tol = 0.01,
            avoid_names = [], 
            end_effector = "panda_hand",
            q_vel_max = np.array([150.0, 150.0, 150.0, 150.0, 180.0, 180.0, 180.0])*np.pi/180.0):
            
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
        self.trajectory = trajectory
        

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
        self.progs = []
        self.qs = []
        self.times = []

        #add initial and final kinematic constraints 
        self.p_tol = np.ones(3)*p_tol
        self.theta_tol = theta_tol

        for i in range(self.trajectory.length()- 1):
            self.progs.append(MathematicalProgram())
            prog = self.progs[-1]
                    
            N = self.trajectory.divisions[i]
            start = self.trajectory.waypoints[i]
            goal = self.trajectory.waypoints[i+1] 
            self.qs.append(prog.NewContinuousVariables(N + 2, 
                self.num_positions,
                name = "q"))

            q = self.qs[-1]

            # add joint limit constraints
            for row in q:
                prog.AddBoundingBoxConstraint(joint_limits['lower'], joint_limits['upper'], row)


            ps_WG = start.pose.translation()
            pg_WG = goal.pose.translation()
            Rs_WG = start.pose.rotation()
            Rg_WG = goal.pose.rotation()


            self.AddPositionConstraint(prog, q[0], ps_WG - self.p_tol, ps_WG + self.p_tol)
            self.AddPositionConstraint(prog, q[-1], pg_WG - self.p_tol, pg_WG + self.p_tol)
            self.AddOrientationConstraint(prog, q[0], Rs_WG, self.theta_tol)
            self.AddOrientationConstraint(prog, q[-1], Rg_WG, self.theta_tol)

            self.times.append(np.linspace(start.time, goal.time, N + 2))

            if N == 0:
                continue
                
            dt = (goal.time -start.time)/(N+1)

    
            #add constraints on intermediate solutions 
            for j in range(len(q) - 1):
                q_now = q[j]
                q_next = q[j+1]
                # ensure adjacent solutions are near one another
                prog.AddCost((q_next - q_now).dot(q_next - q_now)) 
                v1 = (q_next - q_now)/dt
                # joint velocity limits constraint
                prog.AddLinearConstraint(v1, lb = -q_vel_max, ub = q_vel_max)
                if j < (len(q) - 2):
                    q_next_next = q[j+2]
                    v2 = (q_next_next- q_next)/dt
                    prog.AddCost((v2-v1).dot(v2-v1))


            self.results = []

    def Solve(self):
        for i in range(len(self.progs)):
            prog = self.progs[i]
            if (i > 0):
                prev_soln = self.results[-1].GetSolution(self.qs[i-1][-1]) 
                prog.AddLinearEqualityConstraint(self.qs[i][0], prev_soln)
                prog.SetInitialGuess(self.qs[i][0], prev_soln)
            self.results.append(Solve(prog))
        return self.results 



    def get_q_traj(self):
        assert len(self.results) > 0
        q_list = []
        for i in range(len(self.results)):
            if i == 0:
                t_lst = self.times[i]
                q_knots = self.results[i].GetSolution(self.qs[i])
            else:
                t_lst = np.concatenate((t_lst, self.times[i][1:]))
                q_knots = np.concatenate((q_knots, self.results[i].GetSolution(self.qs[i])[1:]))

        #print(q_knots)
        return PiecewisePolynomial.CubicShapePreserving(t_lst, q_knots[:, 0:self.num_positions].T)


    def AddJointCenteringCost(self, q_nominal):
        for i in range(len(self.progs)):
            prog = self.progs[i]
            q = self.qs[i]
            for j in range(len(q)):
                diff = (q[j] - q_nominal)#[1:] # we don't care about base joint
                prog.AddCost(diff.dot(diff))
                        
                       
        
    def AddPositionConstraint(self, prog, q, p_WQ_lower, p_WQ_upper, p_LQ = np.zeros(3)):
        """ Adds a bounding box constraint on the position of the panda's hand

        p_WQ_lower: the lower coordinate of the bounding box in the world frame (np.array (1,3))
        p_WQ_upper: the upper coordinate of the bounding box in the world frame (np.array (1,3))
        p_LQ: an optional offset from the hand frame to the point Q  (np.array (1,3))
        """
        p_WL = lambda q: self.X_WL(q).translation() + p_LQ 
        prog.AddConstraint(p_WL, lb = p_WQ_lower, ub = p_WQ_upper, vars = q)

    def AddOrientationConstraint(self, prog, q, R_WD, theta_tol):
        diff = lambda q: [self.AngleBetween(q, R_WD)]
        prog.AddConstraint(diff, lb = [-theta_tol], ub = [theta_tol], vars = q)

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
