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
    for joint_index in plant.GetJointIndices(panda_model_instance):
        joint = plant.get_mutable_joint(joint_index)
        if isinstance(joint, pydrake.multibody.tree.RevoluteJoint):
            joint.set_default_angle(q0[index])
            index+=1

    return panda_model_instance


def AddPandaHand(plant, panda_model_instance, roll = 0, welded = False):
    """Adds a hand to the panda arm (panda_link8)

    plant: the multibody plant 
    panda_model_instance: the panda model instance to add the hand to
    roll: the rotation of the hand relative to panda_link8
    welded: if we want the version with welded fingers (for control)
    """
    parser = pydrake.multibody.parsing.Parser(plant)

    if welded:
    #TODO(ben): fix this so it works in general
        gripper = parser.AddModelFromFile("/home/" + os.environ["USER"] + "/workspace/PandaStation/models/welded_panda_hand.urdf") 
    else:
        gripper = parser.AddModelFromFile(pydrake.common.FindResourceOrThrow("drake/manipulation/models/franka_description/urdf/hand.urdf"))

    X_8G = RigidTransform(RollPitchYaw(0, 0, roll), [0,0,0])
    plant.WeldFrames(plant.GetFrameByName("panda_link8", panda_model_instance), plant.GetFrameByName("panda_hand",gripper), X_8G)
    return gripper

def AddShape(plant, shape, name, mass = 1, mu = 1, color = [0.5, 0.5, 0.9, 1]):
    """Adds a shape model instance to the multibody plant

    plant: the multibody plant to add the shape to
    shape: the type of shape (pydrake.geometry.{Box, Cylinder, Sphere})
    name: the name of the model instance to add to the multibody plant
    mass: the mass of the shape
    mu: the frictional coefficient of the shape
    color: the color of the shape
    """
    instance = plant.AddModelInstance(name)

    if isinstance(shape, pydrake.geometry.Box):
        inertia = pydrake.multibody.tree.UnitIntertia.SolidBox(
            shape.width(), shape.depth(), shape.height())
    elif isinstance(shape, pydrake.geometry.Cylinder):
        inertia = pydrake.multibody.tree.UnitInertia.SolidCylinder(
            shape.radius(), shape.length())
    elif isinstance(shape, pydrake.geometry.Sphere):
        inertia = pydrake.multibody.tree.UnitInertia.SolidSphere(shape.radius())
    else:
        raise RunTimeError(f"Improper shape type {shape}")
    body = plant.AddRigidBody(
        name, instance,
        pydrake.multibody.tree.SpatialInertia(mass = mass, 
                                              p_PScm_E = np.array([0.,0.,0.]),
                                              G_SP_E = inertia))

    if plant.geometry_source_is_registered():
        if isinstance(shape, pydrake.geometry.Box):
            plant.RegisterCollisionGeometry(
                body, RigidTransform(),
                pydrake.geometry.Box(shape.width() - 0.001,
                                     shape.depth() - 0.001,
                                     shape.height() - 0.001), name,
                pydrake.multibody.plant.CoulombFriction(mu,mu))
            i = 0
            # if it is a box, add sphere at the corners for collisions 
            for x in [-shape.width()/2.0, shape.width()/2.0]:
                for y in [-shape.depth()/2.0, shape.depth()/2.0]:
                    for z in [-shape.height()/2.0, shape.height()/2.0]:
                        plant.RegisterCollisionGeometry(
                            body, RigidTransform([x,y,z]),
                            pydrake.geometry.Sphere(radius = 1e-7),
                            f"contact_sphere{i}",
                            pydrake.multibody.plant.CoulombFriction(mu,mu))
                        i+=1
        else:
            plant.RegisterCollisionGeometry(
                body, RigidTransform(), shape, name,
                pydrake.multibody.plant.CoulombFriction(mu, mu))

        plant.RegisterVisualGeometry(body, RigidTransform(), shape, name, color)

        return instance





