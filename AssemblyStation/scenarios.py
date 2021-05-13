"""
Helper utilities to build scenarios/experiments
"""
import numpy as np
import os

import pydrake.all


from pydrake.all import RigidTransform, RollPitchYaw

def AddPanda(plant, q0 = [0.0, 0.1, 0, -1.2, 0, 1.6, 0], X_WB =  RigidTransform()):
    """ Adds a franka panda arm without any hand to the mutlibody plant and welds it to the world frame


    plant: the multibody plant to add the panda to
    q0: the initial joint positions (optional)
    X_WB: the desired transformation between the world frame (W) and the base link of the panda (B)
    """
    urdf_file = pydrake.common.FindResourceOrThrow("drake/manipulation/models/franka_description/urdf/panda_arm.urdf")

    parser = pydrake.multibody.parsing.Parser(plant)
    panda_model_instance = parser.AddModelFromFile(urdf_file)
    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("panda_link0"), X_WB)
    
    index = 0
    for joint_index in plant.GetJointIndicies(panda_model_instance):
        joint = plant.get_mutable_joint(joint_index)
        if isinstance(joint, pydrake.multibody.tree.RevoluteJoint):
            joint.set_default_angle(q0[index])
            index+=1

    return panda_model_instance

def AddPandaHand(plant, panda_model_instance, roll = 0):
    """Adds a hand to the panda arm (panda_link8)

    plant: the multibody plant 
    panda_model_instance: the panda model instance to add the hand to
    roll: the rotation of the hand relative to panda_link8
    """

    gripper = parser.AddModelFromFile(pydrake.common.FindResourceOrThrow("drake/manipulation/models/panda_description/urdf/panda_hand.urdf"))

    X_8G = RigidTransform(RollPitchYaw(0, 0, roll), [0,0,0])
    plant.WeldFrames(plant.GetFrameByName("panda_link8", panda_model_instance), plant.GetFrameByName("panda_hand",gripper), X_8G)
    return gripper





