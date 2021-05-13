import pydrake.all
from scenarios import AddPanda, AddPandaHand

def MakePandaStation(time_step = 0.002):
    """Constructs a PandaStation"""

    builder = pydrake.systems.framework.DiagramBuilder()

    plant, scene_graph = pydrake.multibody.plant.AddMultibodyPlantSceneGraph(
        builder, time_step = time_step)
    panda = AddPanda(plant)
    hand = AddPandaHand(plant)
    plant.Finalize()

    num_panda_positions = plant.num_positions(panda)

    # add a "pass through" to the system. A pass through is input -> output
    panda_position = builder.AddSystem(pydrake.systems.primitives.PassThrough(num_panda_positions))
    builder.ExportInput(panda_position.get_input_port(), "panda_position")
    builder.ExportOutput(panda_position.get_output_port(), "panda_position_command")

    # export panda state outputs 
    # demux with inputs for panda joint positions and velocities (panda state)
    demux = builder.AddSystem(pydrake.systems.primitives.Demultiplexer(
        2 * num_panda_positions, num_panda_positions))
    builder.Connect(plant.get_state_output_port(panda), demux.get_input_port())
    # sticking to naming conventions from manipulation station for ports
    builder.ExportOutput(demux.get_output_port(0), "panda_position_measured") 
    builder.ExportOutput(demux.get_output_port(1), "panda_velocity_estimated")
    builder.ExportOutput(plant.get_state_output_port(panda), "panda_state_estimated")

    # plant for the panda controller
    controller_plant = pydrake.multibody.plant.MultibodyPlant(time_step = time_step)
    controller_panda = AddPanda(controller_plant)
    AddPandaHand(controller_plant, controller_panda, welded = True) # welded so the controller doesn't care about the hand joints
    controller_plant.Finalize()

    # add panda controller
    panda_controller = builder.AddSystem(
        pydrake.systems.controllers.InverseDynamicsController(
            controller_plant,
            kp =[100]*num_panda_positions,
            ki =[1]*num_panda_positions,
            kd =[20]*num_panda_positions,
            has_reference_acceleration = False))

    panda_controller.set_name("panda_controller")
    builder.Connect(plant.get_state_output_port(panda), panda_controller.get_input_port_estimated_state())

    # feedforward torque
    adder = builder.AddSystem(pydrake.systems.primitves.Adder(2, num_panda_positions))
    builder.Connect(panda_controller.get_output_port_control(),
                    adder.get_input_port(0))
    # passthrough to make the feedforward torque optional (default to zero values)
    torque_passthrough = builder.AddSystem(
        pydrake.systems.primitives.PassThrough([0]*num_panda_positions))
    builder.Connect(torque_passthrough.get_output_port(), adder.get_input_port(1))
    builder.ExportInput(torque_passthrough.get_input_port(), "panda_feedforward_torque")
    builder.Connect(adder.get_output_port(), plant.get_actuation_input_port(panda))

    # add a discete derivative to find velocity command based on positional commands
    desired_state_from_position = builder.AddSystem(
        pydrake.systems.primitives.StateInterpolationWithDiscreteDerivative(
            num_panda_positions, time_step, suppress_initial_transient = True))
    desired_state_from_position.set_name("desired_state_from_position")
    builder.Connect(desired_state_from_position.get_output_port(),
                    panda_controller.get_input_port_desired_state())
    builder.Connect(panda_position.get_output_port(), desired_state_from_position.get_input_port())

    # TODO(ben) panda hand controller

    # export cheat ports
    builder.ExportOutput(scene_graph.get_query_output_port(), "geometry_query")
    builder.ExportOutput(plant.get_contact_results_output_port(), "contact_results")
    builder.ExportOutput(plant.get_state_output_port(), "plant_continuous_state")

    diagram = builder.Build()
    return diagram





