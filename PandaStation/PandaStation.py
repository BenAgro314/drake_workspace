import pydrake.all
from pydrake.all import (
        LoadModelDirectives, ProcessModelDirectives, GeometrySet, Parser,
        DiagramBuilder, AddMultibodyPlantSceneGraph, Diagram, MultibodyPlant,
        Demultiplexer, InverseDynamicsController, Adder, PassThrough,    
        StateInterpolatorWithDiscreteDerivative)
from .scenarios import AddPanda, AddPandaHand, AddRgbdSensors
from .panda_hand_position_controller import PandaHandPositionController, MakeMultibodyStateToPandaHandStateSystem
from .utils import FindResource, AddPackagePaths
import numpy as np
import os

def deg_to_rad(deg):
    return deg*np.pi/180.0

class PandaStation(Diagram):

    def __init__(self, time_step = 0.002):

        Diagram.__init__(self)

        self.time_step = time_step

        self.builder = DiagramBuilder()
        self.plant, self.scene_graph = AddMultibodyPlantSceneGraph(
                                                        self.builder, time_step = self.time_step)
        self.plant.set_name("plant")
        self.set_name("panda_station")
        self.object_ids = []
        self.object_poses = []
        self.camera_info = {} #dictionary in the form name: pose

        self.body_info = {} # path: (name, body_index)

        self.controller_plant = MultibodyPlant(time_step = self.time_step)
        self.welded_hand = False
        self.directive = None

        # this environment has only the hand, for grasp planning
        self.hand_env = HandEnv()


    def Finalize(self):
        """Constructs a PandaStation"""

        assert self.panda.is_valid(), "No panda model added"
        assert self.hand.is_valid(), "No panda hand model added"

        self.hand_env.Finalize()


        self.plant.Finalize()
        
        for i in range(len(self.object_ids)):
            body_index = self.object_ids[i]
            body = self.plant.get_body(body_index)
            self.plant.SetDefaultFreeBodyPose(body, self.object_poses[i])

        num_panda_positions = self.plant.num_positions(self.panda)

        # add a "pass through" to the system. A pass through is input -> output
        panda_position = self.builder.AddSystem(PassThrough(num_panda_positions))
        self.builder.ExportInput(panda_position.get_input_port(), "panda_position")
        self.builder.ExportOutput(panda_position.get_output_port(), "panda_position_command")

        # export panda state outputs 
        # demux with inputs for panda joint positions and velocities (panda state)
        demux = self.builder.AddSystem(Demultiplexer(
            2 * num_panda_positions, num_panda_positions))
        self.builder.Connect(self.plant.get_state_output_port(self.panda), demux.get_input_port())
        # sticking to naming conventions from manipulation station for ports
        self.builder.ExportOutput(demux.get_output_port(0), "panda_position_measured") 
        self.builder.ExportOutput(demux.get_output_port(1), "panda_velocity_estimated")
        self.builder.ExportOutput(self.plant.get_state_output_port(self.panda), "panda_state_estimated")

        # plant for the panda controller
        controller_panda = AddPanda(self.controller_plant)
        AddPandaHand(self.controller_plant, panda_model_instance = controller_panda, welded = True) # welded so the controller doesn't care about the hand joints
        self.controller_plant.Finalize()

        # add panda controller. TODO(ben): make sure that this controller is realistic
        panda_controller = self.builder.AddSystem(InverseDynamicsController(
                self.controller_plant,
                kp =[100]*num_panda_positions,
                ki =[1]*num_panda_positions,
                kd =[20]*num_panda_positions,
                has_reference_acceleration = False))

        panda_controller.set_name("panda_controller")
        self.builder.Connect(self.plant.get_state_output_port(self.panda), panda_controller.get_input_port_estimated_state())

        # feedforward torque
        adder = self.builder.AddSystem(Adder(2, num_panda_positions))
        self.builder.Connect(panda_controller.get_output_port_control(),
                        adder.get_input_port(0))
        # passthrough to make the feedforward torque optional (default to zero values)
        torque_passthrough = self.builder.AddSystem(
            PassThrough([0]*num_panda_positions))
        self.builder.Connect(torque_passthrough.get_output_port(), adder.get_input_port(1))
        self.builder.ExportInput(torque_passthrough.get_input_port(), "panda_feedforward_torque")
        self.builder.Connect(adder.get_output_port(), self.plant.get_actuation_input_port(self.panda))

        # add a discete derivative to find velocity command based on positional commands
        desired_state_from_position = self.builder.AddSystem(
            StateInterpolatorWithDiscreteDerivative(
                num_panda_positions, self.time_step, suppress_initial_transient = True))
        desired_state_from_position.set_name("desired_state_from_position")
        self.builder.Connect(desired_state_from_position.get_output_port(),
                        panda_controller.get_input_port_desired_state())
        self.builder.Connect(panda_position.get_output_port(), desired_state_from_position.get_input_port())

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

    def GetPandaPosition(self, station_context):
        plant_context = self.GetSubsystemContext(self.plant, station_context)
        return self.plant.GetPositions(plant_context, self.panda)

    def SetPandaPosition(self, station_context, q):
        num_panda_positions = self.plant.num_positions(self.panda)
        assert len(q) == num_panda_positions, "Incorrect size of q, needs to be 7"

        plant_context = self.GetSubsystemContext(self.plant, station_context)
        #plant_state = self.GetMutableSubsystemState(self.plant, state)
        self.plant.SetPositions(plant_context, self.panda, q)

    def SetPandaVelocity(self, station_context, state, v):
        num_panda_positions = self.plant.num_positions(self.panda)
        assert len(v) == num_panda_positions, "Incorrect size of v, needs to be 7"

        plant_context = self.GetSubsystemContext(self.plant, multibody)
        plant_state = self.GetMutableSubsystemState(self.plant, state)
        self.plant.SetVelocities(plant_context, plant_state, self.panda, v)

    def SetHandPosition(self, station_context, state, q):
        
        plant_context = self.GetSubsystemContext(self.plant, station_context)
        plant_state = self.GetMutableSubsystemState(self.plant, state)
        self.plant.SetPositions(plant_context, plant_state, self.hand, [q/2.0, q/2.0])

    def SetHandVelocity(self, station_context, state, v):
       
        plant_context = self.GetSubsystemContext(self.plant, station_context)
        plant_state = self.GetMutableSubsystemState(self.plant, state)
        self.plant.SetVelocities(plant_context, plant_state, self.hand, [v/2.0, v/2.0])

    def fix_collisions(self):
        # fix collisions in this model by removing collisions between
        # panda_link5<->panda_link7 and panda_link7<->panda_hand
        panda_link5 =  self.plant.GetFrameByName("panda_link5").body()
        panda_link5 =  GeometrySet(
                self.plant.GetCollisionGeometriesForBody(panda_link5))
        panda_link7 =  self.plant.GetFrameByName("panda_link7").body()
        panda_link7 =  GeometrySet(
                self.plant.GetCollisionGeometriesForBody(panda_link7))
        panda_hand =  self.plant.GetFrameByName("panda_hand").body()
        panda_hand =  GeometrySet(
                self.plant.GetCollisionGeometriesForBody(panda_hand))
        self.scene_graph.ExcludeCollisionsBetween(panda_link5, panda_link7)
        self.scene_graph.ExcludeCollisionsBetween(panda_link7, panda_hand)

    def SetupDefaultStation(self, welded_hand = False):
        self.panda = AddPanda(self.plant)
        self.hand = AddPandaHand(self.plant, panda_model_instance = self.panda, 
                welded = welded_hand)
        self.welded_hand = welded_hand
        self.fix_collisions()
        self.hand_env.AddHand()

    def SetupBinStation(self, welded_hand = False):
        self.directive = FindResource("models/two_bins_w_cameras.yaml")
        parser = Parser(self.plant)
        AddPackagePaths(parser)

        # adds bins and cameras
        ProcessModelDirectives(LoadModelDirectives(self.directive), self.plant, parser)

        self.hand_env.SetupFromFile("models/two_bins_w_cameras.yaml")
        # adds hand and arm
        self.SetupDefaultStation(welded_hand) 
    
    def SetupTableStation(self, welded_hand = False):
        self.directive = FindResource("models/table_top.yaml")
        parser = Parser(self.plant)
        AddPackagePaths(parser)

        # adds bins and cameras
        ProcessModelDirectives(LoadModelDirectives(self.directive), self.plant, parser)

        self.hand_env.SetupFromFile("models/table_top.yaml")
        # adds hand and arm
        self.SetupDefaultStation(welded_hand) 

    def SetupStationFromFile(setup_name, file_name):
        """ setup a the station from a yaml file in the models directory """

        self.setup = setup_name
        
        # TODO(ben): generalize this 
        self.directive = FindResource("models/" + file_name + ".yaml")
        parser = Parser(self.plant)
        AddPackagePaths(parser)

        # adds bins and cameras
        ProcessModelDirectives(LoadModelDirectives(self.directive), self.plant, parser)

        self.hand_env.SetupFromFile("models/" + file_name + ".yaml")
        # adds hand and arm
        self.SetupDefaultStation() 

    def AddModelFromFile(self, path, X_WO, name = None, to_hand_env = False):
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
        if to_hand_env:
            self.hand_env.AddModelFromFile(path, X_WO, name)

    def get_hand_env(self):
        return self.hand_env

    def get_camera_info(self):
        return self.camera_info

    def GetPanda(self):
        return self.panda

    def GetHand(self):
        return self.hand

class HandEnv(Diagram):
    
    def __init__(self, time_step = 0.001):
        Diagram.__init__(self)
        self.time_step = time_step
        self.builder = DiagramBuilder()
        self.plant, self.scene_graph = AddMultibodyPlantSceneGraph(self.builder, time_step = self.time_step)
        self.set_name("hand_env")
        self.plant.set_name("plant")
        self.hand = None   
        self.object_ids = []
        self.object_poses = []
        
    def Finalize(self):
        assert self.hand.is_valid(), "You forgot to setup the environment"
        self.plant.Finalize()
        
        for i in range(len(self.object_ids)):
            body_index = self.object_ids[i]
            body = self.plant.get_body(body_index)
            self.plant.SetDefaultFreeBodyPose(body, self.object_poses[i])
        
        AddRgbdSensors(self.builder, self.plant, self.scene_graph)
        self.builder.ExportOutput(self.scene_graph.get_query_output_port(), "query_object")
        self.builder.BuildInto(self)

    def SetupFromFile(self, filename):
        parser = Parser(self.plant)
        AddPackagePaths(parser)
        directive = FindResource(filename)
        ProcessModelDirectives(LoadModelDirectives(directive), self.plant, parser)
        
    def AddModelFromFile(self, path, X_WO, name = None):
        parser = Parser(self.plant)
        if name is None:
            num = str(len(self.object_ids))
            name = "added_model_" + num
        model = parser.AddModelFromFile(path, name)
        indices = self.plant.GetBodyIndices(model)
        assert len(indices) == 1, "Currently, we only support adding models with one body"
        self.object_ids.append(indices[0])
        self.object_poses.append(X_WO)
        
    def AddHand(self):
        self.hand = AddPandaHand(self.plant, welded = True)
        
    def get_multibody_plant(self):
        return self.plant
    
    def get_scene_graph(self):
        return self.scene_graph
