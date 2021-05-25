import numpy as np
import open3d as o3d
import time
from ompl import base as ob
from ompl import geometric as og
from PandaStation import (
    PandaStation, FindResource, AddPackagePaths, AddRgbdSensors, 
    draw_points, draw_open3d_point_cloud, create_open3d_point_cloud) 
from PandaInverseKinematics import PandaInverseKinematics
from pydrake.all import (
        AbstractValue, LeafSystem, Isometry3, AngleAxis, RigidTransform,
        DiagramBuilder, RotationMatrix, Solve, BasicVector, PiecewisePolynomial,
        InverseKinematics
        )
import pydrake.perception as mut

class BinPointCloudToGraspSystem(LeafSystem):

    def __init__(self):
        LeafSystem.__init__(self)

        self.nq = 7

        camera0_pcd = self.DeclareAbstractInputPort(
                "camera0_pcd", AbstractValue.Make(mut.PointCloud()))
        camera1_pcd = self.DeclareAbstractInputPort(
                "camera1_pcd", AbstractValue.Make(mut.PointCloud()))
        camera2_pcd = self.DeclareAbstractInputPort(
                "camera2_pcd", AbstractValue.Make(mut.PointCloud()))
        self.cameras = {
                'camera0': camera0_pcd, 
                'camera1': camera1_pcd, 
                'camera2': camera2_pcd}

        self.panda_position_input_port = self.DeclareVectorInputPort("panda_position", 
                BasicVector(self.nq))
        self.panda_position_command_output_port = self.DeclareVectorOutputPort(
                "panda_position_command", BasicVector(self.nq),
                self.DoCalcOutput)
        self.hand_command_output_port = self.DeclareVectorOutputPort(
                "hand_position_command", BasicVector(1),
                self.CalcHandOutput)

        # create an internal multibody plant model with just the arm and the bins
        # (ie. what the robot knows)
        builder = DiagramBuilder()
        station = builder.AddSystem(PandaStation())
        station.SetupBinStation()
        self.plant = station.get_multibody_plant()
        self.panda = self.plant.GetModelInstanceByName("panda")
        station.Finalize()
        self.station_context = station.CreateDefaultContext()
        self.plant_context = self.plant.GetMyContextFromRoot(self.station_context)
        self.scene_graph = station.get_scene_graph()
        self.scene_graph_context = station.GetSubsystemContext(self.scene_graph, 
                self.station_context)
        self.query_output_port = self.scene_graph.GetOutputPort("query")

        #self.DeclareAbstractOutputPort("X_WG", lambda: AbstractValue.Make(RigidTransform()),
         #       self.DoCalcOutput)

        self.rng = np.random.default_rng()


        # these store the end effector pose and arm joint positions for the 
        # goal grasping pose
        self.X_WG = None
        self.q_G = None
        self.X_GPre = RigidTransform(RotationMatrix(),
                [0, 0, -0.2])
        self.q_Pre = None

        rot = RotationMatrix.MakeZRotation(np.pi)
        rot = rot.multiply(RotationMatrix.MakeXRotation(np.pi))
        self.X_WPlace = RigidTransform( rot ,[.65, 0.1, 0.29])

        self.q_Place = None
        self.q_initial = None

        # ik params
        self.avoid_names = ['bin0', 'bin1'] #TODO(ben): implement/test this
        self.q_nominal = np.array([ 0., 0.55, 0., -1.45, 0., 1.58, 0.])
        self.p_tol = 0.01*np.ones(3)
        self.theta_tol = 0.01

        self.avoid_geom_ids = self.get_avoid_geom_ids() 
        '''
        the below can be one of:
            - initialization: before finding the first gripper pose
            - to_prepick: moving to the prepick position
            - picking: picking up the manipuland
            - to_place: placing the manipuland
            - placing:
            - to_rest: move to initial position, and then back to initialization
        '''
        self.status = "initialization" 
        self.panda_traj = None
        self.hand_traj = None

    def process_bin_point_cloud(self, context):

        # Compute crop box.
        bin_instance = self.plant.GetModelInstanceByName('bin0')
        bin_body = self.plant.GetBodyByName("bin_base", bin_instance)
        X_B = self.plant.EvalBodyPoseInWorld(self.plant_context, bin_body)
        margin = 0.001  # only because simulation is perfect!
        a = X_B.multiply([-.22+0.025+margin, -.29+0.025+margin, 0.015+margin])
        b = X_B.multiply([.22-0.1-margin, .29-0.025-margin, 2.0])
        crop_min = np.minimum(a,b)
        crop_max = np.maximum(a,b)

        # Evaluate the camera output ports to get the images.
        merged_pcd = o3d.geometry.PointCloud()
        for name,port in list(self.cameras.items()):
            point_cloud = self.EvalAbstractInput(context, port.get_index()).get_value()
            pcd = create_open3d_point_cloud(point_cloud)

            # Crop to region of interest.
            pcd = pcd.crop(
                o3d.geometry.AxisAlignedBoundingBox(min_bound=crop_min,
                                                    max_bound=crop_max))    

            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=0.1, max_nn=30))

            camera = self.plant.GetModelInstanceByName(name)
            body = self.plant.GetBodyByName("base", camera)
            X_C = self.plant.EvalBodyPoseInWorld(self.plant_context, body)
            pcd.orient_normals_towards_camera_location(X_C.translation())
            
            # Merge point clouds.
            merged_pcd += pcd

        # Voxelize down-sample.  (Note that the normals still look reasonable)
        return merged_pcd.voxel_down_sample(voxel_size=0.005)

    def grasp_candidate_cost(self, X_G, cloud):
        
        body = self.plant.GetBodyByName("panda_hand")

        X_GW = X_G.inverse()
        pts = np.asarray(cloud.points).T
        p_GC = X_GW.multiply(pts)

        # Crop to a region inside of the finger box.
        crop_min = [-0.009, -0.035, 0.06]
        crop_max = [0.009, 0.035, 0.115]
        indices = np.all((crop_min[0] <= p_GC[0,:], p_GC[0,:] <= crop_max[0],
                          crop_min[1] <= p_GC[1,:], p_GC[1,:] <= crop_max[1],
                          crop_min[2] <= p_GC[2,:], p_GC[2,:] <= crop_max[2]), 
                         axis=0)


        query_object = self.scene_graph.get_query_output_port().Eval(
                self.scene_graph_context)
        
        # check collisions with objects in the world 
        #if query_object.HasCollisions():
            #print('collision with world')
            #return np.inf

        # Check collisions between the gripper and the point cloud
        margin = 0.0  # must be smaller than the margin used in the point cloud preprocessing.
        for pt in cloud.points:
            distances = query_object.ComputeSignedDistanceToPoint(pt, threshold=margin)
            if distances:
                #print('collision with point cloud')
                return np.inf

        n_GC = X_GW.rotation().multiply(np.asarray(cloud.normals)[indices,:].T)

        # Penalize deviation of the gripper from vertical.
        # weight * -dot([0, 0, -1], R_G * [0, 1, 0]) = weight * R_G[2,1]
        cost = 20.0*X_G.rotation().matrix()[2, 1]

        # Reward sum |dot product of normals with gripper x|^2
        cost -= np.sum(n_GC[0,:]**2)

        return cost


    def generate_grasp_candidate_antipodal(self, cloud):
        """
        Picks a random point in the cloud, and aligns the robot finger with the normal of that pixel. 
        The rotation around the normal axis is drawn from a uniform distribution over [min_roll, max_roll].
        """
        body = self.plant.GetBodyByName("panda_hand")

        index = self.rng.integers(0,len(cloud.points)-1)

        # Use S for sample point/frame.
        p_WS = np.asarray(cloud.points[index])
        n_WS = np.asarray(cloud.normals[index])

        if not np.isclose(np.linalg.norm(n_WS), 1.0):
            return np.inf, None, None

        Gy = n_WS # gripper y axis aligns with normal
        # make orthonormal z axis, aligned with world down
        z = np.array([0.0, 0.0, -1.0])
        if np.abs(np.dot(z,Gy)) < 1e-6:
            # normal was pointing straight down.  reject this sample.
            #print('here')
            return np.inf, None, None

        Gz = z - np.dot(z,Gy)*Gy
        Gx = np.cross(Gy, Gz)
        R_WG = RotationMatrix(np.vstack((Gx, Gy, Gz)).T)
        p_GS_G = [0, 0.029, 0.1]

        # Try orientations from the center out
        min_pitch=-np.pi/3.0
        max_pitch=np.pi/3.0
        alpha = np.array([0.5])#, 0.65, 0.35])#, 0.8, 0.2, 1.0, 0.0])
        thetas = (min_pitch + (max_pitch - min_pitch)*alpha)
        warm_start = self.q_nominal
        for theta in thetas: 
            #print(f"trying angle {theta}")
            # Rotate the object in the hand by a random rotation (around the normal).
            R_WG2 = R_WG.multiply(RotationMatrix.MakeYRotation(theta))

            # Use G for gripper frame.
            p_SG_W = - R_WG2.multiply(p_GS_G)
            p_WG = p_WS + p_SG_W 

            X_G = RigidTransform(R_WG2, p_WG)
            
            #ik
            q = self.find_qV2(X_G, warm_start)

            if q is None:
                #print("failed ik")
                continue

            warm_start =  q
            
            self.plant.SetPositions(self.plant_context, self.panda, q)
            self.plant.SetPositions(self.plant_context, self.plant.GetModelInstanceByName("hand"), [-0.04, 0.04])

            cost = self.grasp_candidate_cost(X_G, cloud)
            if np.isfinite(cost):
                return cost, X_G, q

        return np.inf, None, None

    def find_qV2(self, X_G, warm_start =np.array([ 0., 0.55, 0., -1.45, 0., 1.58, 0.])):
        #s = time.time() 
        ik = InverseKinematics(self.plant, self.plant_context)
        ik.AddPositionConstraint(
                self.plant.GetFrameByName("panda_hand"),
                np.zeros(3),
                self.plant.world_frame(),
                X_G.translation() - self.p_tol,
                X_G.translation() + self.p_tol)
        ik.AddOrientationConstraint(
                self.plant.GetFrameByName("panda_hand"),
                RotationMatrix(),
                self.plant.world_frame(),
                X_G.rotation(),
                self.theta_tol)
        ik.AddMinimumDistanceConstraint(0.01, 0.1)
        q = ik.q()
        prog = ik.prog()
        q_nom = np.concatenate((self.q_nominal, np.zeros(2)))
        prog.AddQuadraticErrorCost(np.identity(len(q)), q_nom, q)
        q_warm = np.concatenate((warm_start, np.zeros(2)))
        prog.SetInitialGuess(q, q_warm)

        result = Solve(prog)

        #print(f"ik duration: {time.time() - s}")

        if not result.is_success():
            return None


        return result.GetSolution(q)[:-2]
    
    def find_q(self, X_G):
        """ finds the joint positions q given a desired gripper end effector 
        position in the world frame X_G
        """
        #s = time.time() 
        ik = PandaInverseKinematics(
                self.plant, 
                self.plant_context, 
                self.panda, 
                avoid_names = self.avoid_names)

        p_WQ = X_G.translation()
        ik.AddPositionConstraint(p_WQ+self.p_tol, p_WQ+self.p_tol)
        ik.AddOrientationConstraint(X_G.rotation(), self.theta_tol)
        ik.AddMinDistanceConstraint(0.01)

        prog = ik.get_prog()
        q = ik.get_q()
        prog.AddQuadraticErrorCost(np.identity(len(q)), self.q_nominal, q)
        prog.SetInitialGuess(q, self.q_nominal)

        result = Solve(prog)

        #print(f"ik duration: {time.time() - s}")

        if not result.is_success():
            return None


        return result.GetSolution(q)

    def CalcHandOutput(self, context, output):
        time = context.get_time()
        if (self.hand_traj is None) or (time > self.hand_traj.end_time()):
            # by default keep hand closed
            output.set_value([0])
        else:
            q_command = self.hand_traj.value(time).flatten()
            output.set_value(q_command)


    def DoCalcOutput(self, context, output):

        time = context.get_time()

        if self.status == "initialization":
            print("INITIALIZATION")
            cropped_pcd = self.process_bin_point_cloud(context)

            q_start = self.EvalVectorInput(context, 
                    self.panda_position_input_port.get_index()).get_value()

            self.q_initial = np.copy(q_start)

            cost = np.inf
            q_goal = None
            X_G = None
            #print('looking for candidate grasp')
            for i in range(100):
                print(f"point choice number {i+1}")
                cost, X_G, q_goal = self.generate_grasp_candidate_antipodal(cropped_pcd)
                if np.isfinite(cost):
                    print("FOUND GRASP")
                    self.X_WG = X_G
                    self.q_G = q_goal
                    break
                
            assert np.isfinite(cost), "could not find valid grasp pose"
           
            # pregrasp pose should be above grasping pose
            self.X_WPre, self.q_Pre = self.pregrasp()
            print("pick:", self.X_WG)
            print("prepick:", self.X_WPre)
            self.panda_traj = self.rrt_trajectory(q_start, time, self.q_Pre, time + 10)
            self.hand_traj = self.make_hand_traj(0.08, time, 0.08, time +10)
            self.status = "to_prepick"
            print("TO PREPICK")

        if self.status == "to_prepick":
            if (time <= self.panda_traj.end_time()):
                q_command = self.panda_traj.value(time).flatten()
                output.set_value(q_command)
            else:
                self.status = "picking"
                q_start = self.EvalVectorInput(context, 
                        self.panda_position_input_port.get_index()).get_value()
                self.make_pick_traj(q_start, time)
                print("PICKING")

        if self.status == "picking":
            if (time <= self.panda_traj.end_time()):
                q_command = self.panda_traj.value(time).flatten()
                output.set_value(q_command)
            else:
                X_WPreplace, q_Preplace = self.preplace()
                q_start = self.EvalVectorInput(context, 
                        self.panda_position_input_port.get_index()).get_value()
                self.q_Place  = self.find_qV2(self.X_WPlace)
                assert self.q_Place is not None
                self.panda_traj = self.rrt_trajectory(q_start, 
                        time, q_Preplace, time+10)
                self.hand_traj = self.make_hand_traj(0, time, 0, time+10)
                self.status = "to_preplace"
                print("TO PREPLACE")

        if self.status == "to_preplace":
            if (time <= self.panda_traj.end_time()):
                q_command = self.panda_traj.value(time).flatten()
                output.set_value(q_command)
            else:
                q_start = self.EvalVectorInput(context, 
                        self.panda_position_input_port.get_index()).get_value()
                self.make_place_traj(q_start, time)
                self.status = "placing"
                print("PLACING")

        if self.status == "placing":
            if (time <= self.panda_traj.end_time()):
                q_command = self.panda_traj.value(time).flatten()
                output.set_value(q_command)
            else:
                q_start = self.EvalVectorInput(context, 
                        self.panda_position_input_port.get_index()).get_value()

                self.panda_traj = self.rrt_trajectory(q_start, time, 
                        self.q_initial, time + 4)
                self.hand_traj = self.make_hand_traj(0.08, time, 0.08, time+2)
                self.status = "to_rest"
                print("TO REST")

        if self.status == "to_rest":
            if (time <= self.panda_traj.end_time()):
                q_command = self.panda_traj.value(time).flatten()
                output.set_value(q_command)
            else:
                q_command = self.q_initial
                output.set_value(q_command)

    def make_place_traj(self, q_start, time):
        # hand moving downwards
        self.panda_traj = self.rrt_trajectory(q_start, time, self.q_Place, time + 2)
        self.hand_traj = self.make_hand_traj(0.0, time, 0.0, time+2)

        # hand holding while the gripper is opening
        traj_hold = self.make_hold_traj(self.q_Place, time+2, time+4)
        self.panda_traj.ConcatenateInTime(traj_hold)
        hand_open = self.make_hand_traj(0.0, time+2, 0.08, time+4)
        self.hand_traj.ConcatenateInTime(hand_open)

        # hand raising upwards
        traj_up = self.rrt_trajectory(self.q_Place, time+4, q_start, time+6)
        self.panda_traj.ConcatenateInTime(traj_up)
        hand_opened = self.make_hand_traj(0.08, time+4, 0.08, time+6)
        self.hand_traj.ConcatenateInTime(hand_opened)

    def make_pick_traj(self, q_start, time):
        # hand moving downwards
        self.panda_traj = self.rrt_trajectory(q_start, time, self.q_G, time + 2)
        self.hand_traj = self.make_hand_traj(0.08, time, 0.08, time+2)

        # hand holding while the gripper is closing
        traj_hold = self.make_hold_traj(self.q_G, time+2, time+4)
        self.panda_traj.ConcatenateInTime(traj_hold)
        hand_close = self.make_hand_traj(0.08, time+2, 0, time+4)
        self.hand_traj.ConcatenateInTime(hand_close)

        # hand raising upwards
        traj_up = self.rrt_trajectory(self.q_G, time+4, q_start, time+6)
        self.panda_traj.ConcatenateInTime(traj_up)
        hand_closed = self.make_hand_traj(0, time+4, 0, time+6)
        self.hand_traj.ConcatenateInTime(hand_closed)
    
    @staticmethod
    def make_hold_traj(to_hold, start_time, end_time):
        """ make a trajectory that holds the position to_hold from star_time
        to end_time
        """
        time = np.array([start_time, end_time])
        qs = np.array([to_hold, to_hold])
        return PiecewisePolynomial.ZeroOrderHold(time, qs.T)

    def preplace(self):
        X_WPre= self.X_WPlace.multiply(self.X_GPre)
        q_Pre = self.find_qV2(X_WPre)
        assert q_Pre is not None, "invalid pregrasp pose"
        return X_WPre, q_Pre

    def pregrasp(self):
        X_WPre = self.X_WG.multiply(self.X_GPre)
        q_Pre = self.find_qV2(X_WPre)
        assert q_Pre is not None, "invalid pregrasp pose"
        return X_WPre, q_Pre

    def make_hand_traj(self, q_start, start_time, q_end, end_time):
        # these input q's are floats not lists
        time = np.array([start_time, end_time])
        qs = np.array([[q_start], [q_end]])
        return PiecewisePolynomial.FirstOrderHold(time, qs.T)

    #----------------- RRT -----------------------------------

    def rrt_trajectory(self, q_start, start_time, q_goal, goal_time):

        joint_limits = self.joint_limits() 

        space = ob.RealVectorStateSpace(self.nq)
        bounds = ob.RealVectorBounds(self.nq)

        for i in range(self.nq):
            bounds.setLow(i, joint_limits[i][0])
            bounds.setHigh(i, joint_limits[i][1])

        space.setBounds(bounds)
        ss = og.SimpleSetup(space)
        ss.setStateValidityChecker(ob.StateValidityCheckerFn(self.isStateValid))
        start = self.q_to_state(space, q_start)
        goal = self.q_to_state(space, q_goal)
        ss.setStartAndGoalStates(start, goal) 
        solved = ss.solve()

        assert solved

        ss.simplifySolution()
        path = ss.getSolutionPath()

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
        
        
        if (costs[-1] == 0):
            costs[-1] = 1

        qs = np.array(res)
        times = start_time + ((goal_time - start_time)*np.array(costs)/costs[-1])

        if len(qs) < 3:
            return PiecewisePolynomial.FirstOrderHold(times, 
                    qs[:, 0:self.nq].T)
        
        return PiecewisePolynomial.CubicShapePreserving(times,
                qs[:, 0:self.nq].T)
        

    def joint_limits(self):
        joint_inds = self.plant.GetJointIndices(self.panda)[:self.nq]
        joint_limits = []
        for i in joint_inds:
            joint = self.plant.get_joint(i)
            joint_limits.append((joint.position_lower_limits()[0],
                                 joint.position_upper_limits()[0]))
        return joint_limits

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

    def parse_state(self, state):
        """ parses ompl RealVectorStateSpace::StateType into a numpy array"""
        q = []
        for i in range(self.nq):
            q.append(state[i])
        return np.array(q)

    def q_to_state(self, space, q):
        """ turns a python list q into an ompl state"""
        state = ob.State(space)
        for i in range(len(q)):
            state[i] = q[i]
        return state

    def get_avoid_geom_ids(self):

        avoid_geom_ids = []
        for name in self.avoid_names:
            bodies = self.plant.GetBodyIndices(self.plant.GetModelInstanceByName(name))
            for i in bodies:
                avoid_geom_ids += self.plant.GetCollisionGeometriesForBody(self.plant.get_body(i))
        return avoid_geom_ids

