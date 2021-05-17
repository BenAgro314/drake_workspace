import numpy as np
from pydrake.all import (
        MultibodyPlant, 
        RollPitchYaw, 
        RotationMatrix,
        AutoDiffXd)
from pydrake.solvers.mathematicalprogram import MathematicalProgram, Solve


def resolve_frame(plant, F):
    return plant.GetFrameByName(F.name(), F.model_instance())

def EvalDistance(plant, context, geom_id1, geom_id2):
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
    distance = query_object.ComputeSignedDistancePairClosestPoints(geom_id1, geom_id2).distance
    return distance

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
        if q.dtype == float:
            plant = self.plant_f
            context = self.context_f
        else:
            plant = self.plant_ad
            context = self.context_ad

        plant.SetPositions(context, self.model_instance, q)
        min_dist = 10e9
        for id_arm in self.panda_geom_ids:
            for id_other in self.avoid_geom_ids:
                d = EvalDistance(plant, context, id_arm, id_other)
                min_dist = min(d, min_dist)

        return min_dist
        
        




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

            
