import pydrake.all
from pydrake.all import LoadModelDirectives, ProcessModelDirectives
from .scenarios import AddPanda, AddPandaHand, AddRgbdSensors
from .panda_hand_position_controller import PandaHandPositionController, MakeMultibodyStateToPandaHandStateSystem
from .utils import FindResource, AddPackagePaths
import numpy as np
import os

def deg_to_rad(deg):
    return deg*np.pi/180.0

class PandaStation(pydrake.systems.framework.Diagram):

    def __init__(self, time_step = 0.002):

        pydrake.systems.framework.Diagram.__init__(self)

        self.time_step = 0.002

        self.builder = pydrake.systems.framework.DiagramBuilder()
        self.plant, self.scene_graph = pydrake.multibody.plant.AddMultibodyPlantSceneGraph(
                                                        self.builder, time_step = self.time_step)
        self.setup = None
        self.plant.set_name("plant")
        self.set_name("panda_station")
        self.object_ids = []
        self.object_poses = []
        self.camera_info = {} #dictionary in the form name: pose


    def Finalize(self):
        """Constructs a PandaStation"""

        assert self.panda.is_valid(), "No panda model added"
        assert self.hand.is_valid(), "No panda hand model added"


        self.plant.Finalize()
        
        for i in range(len(self.object_ids)):
            body_index = self.object_ids[i]
            body = self.plant.get_body(body_index)
            self.plant.SetDefaultFreeBodyPose(body, self.object_poses[i])

        num_panda_positions = self.plant.num_positions(self.panda)

        # add a "pass through" to the system. A pass through is input -> output
        panda_position = self.builder.AddSystem(pydrake.systems.primitives.PassThrough(num_panda_positions))
        self.builder.ExportInput(panda_position.get_input_port(), "panda_position")
        self.builder.ExportOutput(panda_position.get_output_port(), "panda_position_command")

        # export panda state outputs 
        # demux with inputs for panda joint positions and velocities (panda state)
        demux = self.builder.AddSystem(pydrake.systems.primitives.Demultiplexer(
            2 * num_panda_positions, num_panda_positions))
        self.builder.Connect(self.plant.get_state_output_port(self.panda), demux.get_input_port())
        # sticking to naming conventions from manipulation station for ports
        self.builder.ExportOutput(demux.get_output_port(0), "panda_position_measured") 
        self.builder.ExportOutput(demux.get_output_port(1), "panda_velocity_estimated")
        self.builder.ExportOutput(self.plant.get_state_output_port(self.panda), "panda_state_estimated")

        # plant for the panda controller
        controller_plant = pydrake.multibody.plant.MultibodyPlant(time_step = self.time_step)
        controller_panda = AddPanda(controller_plant)
        AddPandaHand(controller_plant, controller_panda, welded = True) # welded so the controller doesn't care about the hand joints
        controller_plant.Finalize()

        # add panda controller. TODO(ben): make sure that this controller is realistic
        panda_controller = self.builder.AddSystem(
            pydrake.systems.controllers.InverseDynamicsController(
                controller_plant,
                kp =[100]*num_panda_positions,
                ki =[1]*num_panda_positions,
                kd =[20]*num_panda_positions,
                has_reference_acceleration = False))

        panda_controller.set_name("panda_controller")
        self.builder.Connect(self.plant.get_state_output_port(self.panda), panda_controller.get_input_port_estimated_state())

        # feedforward torque
        adder = self.builder.AddSystem(pydrake.systems.primitives.Adder(2, num_panda_positions))
        self.builder.Connect(panda_controller.get_output_port_control(),
                        adder.get_input_port(0))
        # passthrough to make the feedforward torque optional (default to zero values)
        torque_passthrough = self.builder.AddSystem(
            pydrake.systems.primitives.PassThrough([0]*num_panda_positions))
        self.builder.Connect(torque_passthrough.get_output_port(), adder.get_input_port(1))
        self.builder.ExportInput(torque_passthrough.get_input_port(), "panda_feedforward_torque")
        self.builder.Connect(adder.get_output_port(), self.plant.get_actuation_input_port(self.panda))

        # add a discete derivative to find velocity command based on positional commands
        desired_state_from_position = self.builder.AddSystem(
            pydrake.systems.primitives.StateInterpolatorWithDiscreteDerivative(
                num_panda_positions, self.time_step, suppress_initial_transient = True))
        desired_state_from_position.set_name("desired_state_from_position")
        self.builder.Connect(desired_state_from_position.get_output_port(),
                        panda_controller.get_input_port_desired_state())
        self.builder.Connect(panda_position.get_output_port(), desired_state_from_position.get_input_port())

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

    def get_scene_graph(self):
        return self.scene_graph

    def GetPandaPosition(self, station_context):
        plant_context = self.GetSubsystemContext(self.plant, station_context)
        return self.plant.GetPositions(plant_context, self.panda)

    def SetPandaPosition(self, station_context, state, q):
        num_panda_positions = self.plant.num_positions(self.panda)
        assert len(q) == num_panda_positions, "Incorrect size of q, needs to be 7"

        plant_context = self.GetSubsystemContext(self.plant, station_context)
        plant_state = self.GetMutableSubsystemState(self.plant, state)
        self.plant.SetPositions(plant_context, plant_state, self.panda, q)

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

    def SetupDefaultStation(self):
        self.panda = AddPanda(self.plant)
        self.hand = AddPandaHand(self.plant, self.panda)

    def SetupBinStation(self):
        self.setup = "BinStation"

        directive = FindResource("models/two_bins_w_cameras.yaml")
        parser = pydrake.multibody.parsing.Parser(self.plant)
        AddPackagePaths(parser)

        # adds bins and cameras
        ProcessModelDirectives(LoadModelDirectives(directive), self.plant, parser)

        # adds hand and arm
        self.SetupDefaultStation() 

        '''
        # add first bin
        X_WC = pydrake.math.RigidTransform(
                pydrake.math.RotationMatrix.MakeZRotation(np.pi/2), [-0.145, -0.63, 0.075])
        bin1 = parser.AddModelFromFile(bin_file, "bin1")
        self.plant.WeldFrames(self.plant.world_frame(), self.plant.GetFrameByName('bin_base', bin1), X_WC)

        # add second bin
        X_WC = pydrake.math.RigidTransform(
                pydrake.math.RotationMatrix.MakeZRotation(np.pi), [0.5, -0.1, 0.075])
        bin2 = parser.AddModelFromFile(bin_file, "bin2")
        self.plant.WeldFrames(self.plant.world_frame(), self.plant.GetFrameByName('bin_base', bin2), X_WC)

        # add cameras
        X_Camera = pydrake.math.RigidTransform(
                pydrake.math.RollPitchYaw(deg_to_rad(-150), 0, np.pi/2.0).ToRotationMatrix(),
                [0.3, -0.65, 1])
        camera = parser.AddModelFromFile(camera_file, "camera")
        camera = self.plant.GetBodyByName("base", camera)
        self.camera_info['camera'] = X_Camera
        self.plant.WeldFrames(self.plant.world_frame(), camera.body_frame(), X_Camera)
        '''

        # add default hand and arm


    def AddManipulandFromFile(self, model_file, X_WObject):
        parser = pydrake.multibody.parsing.Parser(self.plant)
        number = str(len(self.object_ids))
        name = "added_model_"+number
        model = parser.AddModelFromFile(pydrake.common.FindResourceOrThrow(model_file), name)
        indices = self.plant.GetBodyIndices(model)
        # TODO(ben): generalize this to add any model
        assert len(indices) == 1, "Curently, we only support manipulands with one body"
        self.object_ids.append(indices[0])
        self.object_poses.append(X_WObject)

    def get_camera_info(self):
        return self.camera_info
