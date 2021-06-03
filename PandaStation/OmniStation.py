import pydrake.all
from pydrake.all import (
        LoadModelDirectives, ProcessModelDirectives, GeometrySet, Parser,
        DiagramBuilder, AddMultibodyPlantSceneGraph, Diagram, MultibodyPlant,
        Demultiplexer, InverseDynamicsController, Adder, PassThrough,    
        StateInterpolatorWithDiscreteDerivative, RigidTransform, Parser)
from .scenarios import *
from .panda_hand_position_controller import PandaHandPositionController, MakeMultibodyStateToPandaHandStateSystem
from .utils import FindResource, AddPackagePaths
import numpy as np

def AddOmni(plant, q0 = [0, 0, 0.5, 0.01, np.pi, 0.01], X_WB = RigidTransform()):

    urdf_file = FindResource("models/omni_arm.urdf")

    parser = Parser(plant)
    model = parser.AddModelFromFile(urdf_file)
    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("base_link", model), X_WB)
    
    index = 0
    for joint_index in plant.GetJointIndices(model):
        joint = plant.get_mutable_joint(joint_index)
        if isinstance(joint, pydrake.multibody.tree.RevoluteJoint):
            joint.set_default_angle(q0[index])
        if isinstance(joint, pydrake.multibody.tree.PrismaticJoint):
            joint.set_default_translation(q0[index])
        index+=1

    return model

def AddOmniHand(plant, model_instance, welded = False):
    parser = Parser(plant)

    if welded:
        gripper = parser.AddModelFromFile(FindResource("models/welded_panda_hand.urdf"))
    else:
        gripper = parser.AddModelFromFile(
                FindResource("models/panda_hand_fixed_collisions.urdf"))

    X_YG = RigidTransform()
    plant.WeldFrames(plant.GetFrameByName("yaw_link", model_instance), plant.GetFrameByName("panda_hand",gripper), X_YG)

    return gripper

class OmniStation(Diagram):

    def __init__(self, time_step = 0.002):
        Diagram.__init__(self)
        self.time_step = time_step
        self.builder = DiagramBuilder()
        self.plant, self.scene_graph = AddMultibodyPlantSceneGraph(
                                                        self.builder, time_step = self.time_step)
        self.plant.set_name("plant")
        self.set_name("omni_station")
        self.object_ids = []
        self.object_poses = []
        self.camera_info = {} #dictionary in the form name: pose
        self.body_info = {} # path: (name, body_index)
        self.controller_plant = MultibodyPlant(time_step = self.time_step)
        self.welded_hand = False
        self.directive = None
        self.omni = None
        self.hand = None

    def Finalize(self):

        assert self.omni.is_valid(), "No omni model added"
        assert self.hand.is_valid(), "No hand model added"

        self.plant.Finalize()
        
        for i in range(len(self.object_ids)):
            body_index = self.object_ids[i]
            body = self.plant.get_body(body_index)
            self.plant.SetDefaultFreeBodyPose(body, self.object_poses[i])

        nq = self.plant.num_positions(self.omni)

        omni_position = self.builder.AddSystem(PassThrough(nq))
        self.builder.ExportInput(omni_position.get_input_port(), "omni_position")
        self.builder.ExportOutput(omni_position.get_output_port(), "omni_position_command")

        demux = self.builder.AddSystem(Demultiplexer(
            2 * nq, nq))
        self.builder.Connect(self.plant.get_state_output_port(self.omni), demux.get_input_port())
        self.builder.ExportOutput(demux.get_output_port(0), "omni_position_measured") 
        self.builder.ExportOutput(demux.get_output_port(1), "omni_velocity_estimated")
        self.builder.ExportOutput(self.plant.get_state_output_port(self.omni), "omni_state_estimated")

        controller_omni = AddOmni(self.controller_plant)
        AddOmniHand(self.controller_plant, controller_omni, welded = True)
        self.controller_plant.Finalize()

        omni_controller = self.builder.AddSystem(InverseDynamicsController(
                self.controller_plant,
                kp =[100]*nq,
                ki =[1]*nq,
                kd =[20]*nq,
                has_reference_acceleration = False))

        omni_controller.set_name("omni_controller")
        self.builder.Connect(self.plant.get_state_output_port(self.omni), omni_controller.get_input_port_estimated_state())

        adder = self.builder.AddSystem(Adder(2, nq))
        self.builder.Connect(omni_controller.get_output_port_control(),
                        adder.get_input_port(0))
        torque_passthrough = self.builder.AddSystem(
            PassThrough([0]*nq))
        self.builder.Connect(torque_passthrough.get_output_port(), adder.get_input_port(1))
        self.builder.ExportInput(torque_passthrough.get_input_port(), "omni_feedforward_torque")
        self.builder.Connect(adder.get_output_port(), self.plant.get_actuation_input_port(self.omni))

        desired_state_from_position = self.builder.AddSystem(
            StateInterpolatorWithDiscreteDerivative(
                nq, self.time_step, suppress_initial_transient = True))
        desired_state_from_position.set_name("desired_state_from_position")
        self.builder.Connect(desired_state_from_position.get_output_port(),
                        omni_controller.get_input_port_desired_state())
        self.builder.Connect(omni_position.get_output_port(), desired_state_from_position.get_input_port())

        if not self.welded_hand: # no need to do this if the hands are welded
            #TODO(ben): make sure this hand controller is accurate
            hand_controller = self.builder.AddSystem(PandaHandPositionController())
            hand_controller.set_name("hand_controller")
            self.builder.Connect(hand_controller.GetOutputPort("generalized_force"),             
                            self.plant.get_actuation_input_port(self.hand))
            self.builder.Connect(self.plant.get_state_output_port(self.hand), hand_controller.GetInputPort("state"))
            self.builder.ExportInput(hand_controller.GetInputPort("desired_position"), "hand_position")
            self.builder.ExportInput(hand_controller.GetInputPort("force_limit"), "hand_force_limit")
            hand_mbp_state_to_hand_state = self.builder.AddSystem(
                                                    MakeMultibodyStateToPandaHandStateSystem())
            self.builder.Connect(self.plant.get_state_output_port(self.hand), hand_mbp_state_to_hand_state.get_input_port())
            self.builder.ExportOutput(hand_mbp_state_to_hand_state.get_output_port(), "hand_state_measured")
            self.builder.ExportOutput(hand_controller.GetOutputPort("grip_force"), "hand_force_measured")

        # add any cameras

        AddRgbdSensors(self.builder, self.plant, self.scene_graph)

        # export cheat ports
        self.builder.ExportOutput(self.scene_graph.get_query_output_port(), "geometry_query")
        self.builder.ExportOutput(self.plant.get_contact_results_output_port(), "contact_results")
        self.builder.ExportOutput(self.plant.get_state_output_port(), "plant_continuous_state")

        # for visualization

        self.builder.ExportOutput(self.scene_graph.get_query_output_port(), "query_object")

        self.builder.BuildInto(self) 

    def get_multibody_plant(self):
        return self.plant

    def get_controller_plant(self):
        return self.controller_plant

    def get_scene_graph(self):
        return self.scene_graph

    def SetupDefaultStation(self, welded_hand = False):
        self.omni = AddOmni(self.plant)
        self.hand = AddOmniHand(self.plant, self.omni, welded = welded_hand)
        self.welded_hand = welded_hand

    def SetupBinStation(self, welded_hand = False):
        self.directive = FindResource("models/two_bins_w_cameras.yaml")
        parser = Parser(self.plant)
        AddPackagePaths(parser)

        # adds bins and cameras
        ProcessModelDirectives(LoadModelDirectives(self.directive), self.plant, parser)

        # adds hand and arm
        self.SetupDefaultStation(welded_hand) 
    
    def SetupTableStation(self, welded_hand = False):
        self.directive = FindResource("models/table_top.yaml")
        parser = Parser(self.plant)
        AddPackagePaths(parser)

        # adds bins and cameras
        ProcessModelDirectives(LoadModelDirectives(self.directive), self.plant, parser)

        # adds hand and arm
        self.SetupDefaultStation(welded_hand) 

    def SetupStationFromFile(setup_name, file_name):
        """ setup a the station from a yaml file in the models directory """

        self.setup = setup_name
        
        self.directive = FindResource(file_name)
        parser = Parser(self.plant)
        AddPackagePaths(parser)

        # adds bins and cameras
        ProcessModelDirectives(LoadModelDirectives(self.directive), self.plant, parser)

        # adds hand and arm
        self.SetupDefaultStation() 

    def AddModelFromFile(self, path, X_WO, name = None):
        parser = Parser(self.plant)
        if name is None:
            num = str(len(self.object_ids))
            name = "added_model_" + num
        model = parser.AddModelFromFile(path, name)
        indices = self.plant.GetBodyIndices(model)
        assert len(indices) == 1, "Currently, we only support adding models with one body"
        self.body_info[path] = (name, indices[0])
        self.object_ids.append(indices[0])
        self.object_poses.append(X_WO)

    def get_camera_info(self):
        return self.camera_info

    def GetOmni(self):
        return self.omni

    #TODO(ben): change the naming convention here
    def GetHand(self):
        return self.hand
