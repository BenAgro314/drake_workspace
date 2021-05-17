import pydrake.all
import numpy as np
from pydrake.all import (
        FindResourceOrThrow,
        MathematicalProgram,
        Solve,
        MultibodyPlant,
        Parser,
        DiagramBuilder,
        RigidTransform,
        RollPitchYaw,
        AddMultibodyPlantSceneGraph,
        ConnectMeshcatVisualizer,
        RotationMatrix,
        RollPitchYaw)
from PandaInverseKinematics import *
from meshcat.servers.zmqserver import start_zmq_server_as_subprocess
proc, zmq_url, web_url = start_zmq_server_as_subprocess(server_args=[])


def main():

    panda_file = FindResourceOrThrow("drake/manipulation/models/franka_description/urdf/panda_arm.urdf")
    hand_file = FindResourceOrThrow("drake/manipulation/models/franka_description/urdf/hand.urdf")
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=1e-4)
    parser = Parser(plant)

    panda = parser.AddModelFromFile(panda_file, "panda")
    parser.AddModelFromFile(hand_file, "hand")

    plant.WeldFrames(
        frame_on_parent_P=plant.world_frame(),
        frame_on_child_C =plant.GetFrameByName("panda_link0"),
        X_PC = RigidTransform(RollPitchYaw(0,0,0), np.array([0,0,0])))

    plant.WeldFrames(
        frame_on_parent_P=plant.GetFrameByName("panda_link8"),
        frame_on_child_C=plant.GetFrameByName("panda_hand"),
        X_PC=RigidTransform(RollPitchYaw(0, 0, 0), np.array([0, 0, 0])))

    plant.Finalize()

    diagram = builder.Build()
    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyContextFromRoot(context)

    ik = PandaInverseKinematics(plant, plant_context, panda)
    ik.AddPositionConstraint(np.array([0.3,0.5,0.3]), np.array([0.31, 0.51,0.31]))
    desired = RotationMatrix(RollPitchYaw(0.0, np.pi/2.0, np.pi/2.0))
    ik.AddOrientationConstraint(desired, 0.01)
    prog = ik.get_prog()
    q = ik.get_q()
    result = Solve(prog)

    print(result.GetSolution(q))



if __name__ == "__main__":
    main()




