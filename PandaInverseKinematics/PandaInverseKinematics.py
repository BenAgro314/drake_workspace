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


def resolve_frame(plant, F):
    return plant.GetFrameByName(F.name(), F.model_instance())

def EvalDistance(plant, context, geom_id1, geom_id2, q, lower_ind, num_positions):
    """ finds the distance between the geometries with id geom_id1 and geom_id2 """
    query_port = plant.get_geometry_query_input_port()

    if(not query_port.HasValue(context)):
        print('''MinimumDistanceConstraint: 
            Cannot get a valid geometry::QueryObject. 
            Either the plant geometry_query_input_port() 
            is not properly connected to the SceneGraph's 
            output port, or the plant_context_ is incorrect. 
            Please refer to AddMultibodyPlantSceneGraph on 
            connecting MultibodyPlant to SceneGraph.''')

    query_object = query_port.Eval(context)
    signed_distance_pair = query_object.ComputeSignedDistancePairClosestPoints(geom_id1, geom_id2)
    inspector = query_object.inspector()
    frame_A_id = inspector.GetFrameId(signed_distance_pair.id_A)
    frame_B_id = inspector.GetFrameId(signed_distance_pair.id_B)
    frameA = plant.GetBodyFromFrameId(frame_A_id).body_frame()
    frameB = plant.GetBodyFromFrameId(frame_B_id).body_frame()
    p_ACa = np.array(signed_distance_pair.p_ACa)
    #print(p_ACa, type(p_ACa))
    return CalcDistanceDerivatives(
            plant, 
            context,
            frameA,
            frameB,
            inspector.GetPoseInFrame(signed_distance_pair.id_A).multiply(p_ACa),
            signed_distance_pair.distance,
            signed_distance_pair.nhat_BA_W,
            q,
            lower_ind,
            num_positions)


def CalcDistanceDerivatives(plant, context, frameA, frameB, p_ACa, distance, nhat_BA_W, q, lower_ind, num_positions):
    Jq_v_BCa_W = plant.CalcJacobianTranslationalVelocity(
            context, 
            JacobianWrtVariable.kQDot, 
            frameA, 
            p_ACa, 
            frameB, 
            plant.world_frame())

    Jq_v_BCa_W = Jq_v_BCa_W[:, lower_ind:lower_ind+num_positions]
    ddistance_dq = nhat_BA_W.transpose().dot(Jq_v_BCa_W)

    distance_ad = AutoDiffXd(distance,
            ddistance_dq.dot(autoDiffToGradientMatrix(q)))

    return distance_ad


class PandaInverseKinematics:


    def __init__(self, plant, plant_context, model_instance, avoid_names = [], end_effector = "panda_hand"):
        self.plant_f = plant
        self.context_f = plant_context
        test = plant.get_geometry_query_input_port()
        self.plant_ad = self.plant_f.ToAutoDiffXd()
        #print(test.HasValue(plant_context))
        #print(self.plant_f.geometry_source_is_registered())
        #print(self.plant_ad.geometry_source_is_registered())
        self.context_ad = self.plant_ad.CreateDefaultContext()
        self.context_ad.SetTimeStateAndParametersFrom(self.context_f)
        #print(self.plant_ad.get_geometry_query_input_port().HasValue(self.context_ad))
        #print(self.plant_f.get_geometry_query_input_port().HasValue(self.context_f))
        self.model_instance = model_instance
        self.respect_joint_limits = True

        self.W = self.plant_f.world_frame()
        self.L = self.plant_f.GetFrameByName(end_effector)

        self.prog = MathematicalProgram()
        self.num_positions = plant.num_positions(self.model_instance)
        self.q = self.prog.NewContinuousVariables(self.num_positions)

        # add constraint to respect joint limits 
        joint_indices = plant.GetJointIndices(model_instance)[:self.num_positions] # remove the last joint because they are fixed
        joint_limits = {'lower': [], 'upper': []}
        for ind in joint_indices:
            joint = plant.get_joint(ind)
            joint_limits['lower'].append(joint.position_lower_limits()[0])
            joint_limits['upper'].append(joint.position_upper_limits()[0])
        self.prog.AddBoundingBoxConstraint(joint_limits['lower'], joint_limits['upper'], self.q)
        # we want to find the indicies of the generalized positions of the arm in all of the generalized positions (they will be contiguous in the array)
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
 

    def AddPositionConstraint(self, p_WQ_lower, p_WQ_upper, p_LQ = np.zeros(3)):
        """ Adds a bounding box constraint on the position of the panda's hand

        p_WQ_lower: the lower coordinate of the bounding box in the world frame (np.array (1,3))
        p_WQ_upper: the upper coordinate of the bounding box in the world frame (np.array (1,3))
        p_LQ: an optional offset from the hand frame to the point Q  (np.array (1,3))
        """
        p_WL = lambda q: self.X_WL(q).translation() + p_LQ 
        self.prog.AddConstraint(p_WL, lb = p_WQ_lower, ub = p_WQ_upper, vars = self.q)

    def AddOrientationConstraint(self, R_WD, theta_tol):
        diff = lambda q: [self.AngleBetween(q, R_WD)]
        self.prog.AddConstraint(diff, lb = [-theta_tol], ub = [theta_tol], vars = self.q)

    def AddMinDistanceConstraint(self, d_min):
        min_dist = lambda q: [self.MinDistance(q)]
        self.prog.AddConstraint(min_dist, lb = [d_min], ub = [100], vars = self.q)

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
        R_LW = plant.CalcRelativeTransform(context, resolve_frame(plant, self.L), resolve_frame(plant, self.W)).rotation()
        R_LD = R_LW.matrix().dot(R_WD.matrix())
        theta = np.arccos(0.5*(R_LD.trace() - 1))
        return abs(theta)

    def MinDistance(self, q):
        self.plant_f.SetPositions(self.context_f, self.model_instance, autoDiffToValueMatrix(q))
        distances = []
        for id_arm in self.panda_geom_ids:
            for id_other in self.avoid_geom_ids:
                d = EvalDistance(self.plant_f, self.context_f, id_arm, id_other, q, self.lower_ind, self.num_positions)
                distances.append(d)
        return min(distances)
        

    def X_WL(self, q):
        if q.dtype == float:
            plant = self.plant_f
            context = self.context_f
        else:
            plant = self.plant_ad
            context = self.context_ad
        plant.SetPositions(context, self.model_instance, q)
        return plant.CalcRelativeTransform(context, resolve_frame(plant, self.W), resolve_frame(plant, self.L))


                
    def get_prog(self):
        return self.prog
    
    def get_q(self):
        return self.q

            
