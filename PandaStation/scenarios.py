"""
Helper utilities to build scenarios/experiments
"""
import numpy as np
import os
from .utils import FindResource
import pydrake.all


from pydrake.all import (
        RigidTransform, RollPitchYaw, MakeRenderEngineVtk, 
        RenderEngineVtkParams, DepthRenderCamera, RenderCameraCore,
        CameraInfo, ClippingRange, DepthRange, ModelInstanceIndex,
        RgbdSensor, DepthImageToPointCloud, BaseField, LeafSystem,
        AbstractValue
        )

def AddRgbdSensors(builder, plant, scene_graph ,
                   also_add_point_clouds=True, model_instance_prefix="camera",
                   depth_camera = None, renderer = None):
    """ Adds a rgbd sensor to each body in the plant with a name starting with 
    body_prefix. If camera_info is None, a default camera info will be used. 
    if renderer is None, then we will assume the name "my_renderer", and create
    a VTK renderer if a renderer of that name does not exist"""

    if not renderer:
        renderer = "my_renderer"

    if not scene_graph.HasRenderer(renderer):
        scene_graph.AddRenderer(renderer, MakeRenderEngineVtk(RenderEngineVtkParams()))

    if not depth_camera:
        depth_camera = DepthRenderCamera(
                RenderCameraCore(
                    renderer, CameraInfo(width = 640, height = 480, fov_y = np.pi/4),
                    ClippingRange(near=0.1, far=10.0), RigidTransform()),
                DepthRange(0.1, 10.0))

    for index in range(plant.num_model_instances()):
        model_instance_index = ModelInstanceIndex(index)
        model_name = plant.GetModelInstanceName(model_instance_index)

        if model_name.startswith(model_instance_prefix):
            body_index = plant.GetBodyIndices(model_instance_index)[0]
            rgbd = builder.AddSystem(
                    RgbdSensor(parent_id=plant.GetBodyFrameIdOrThrow(body_index),
                               X_PB=RigidTransform(), depth_camera=depth_camera,
                               show_window = False))
            rgbd.set_name(model_name)
            builder.Connect(scene_graph.get_query_output_port(), 
                            rgbd.query_object_input_port())
            builder.ExportOutput(rgbd.color_image_output_port(), 
                                 f"{model_name}_rgb_image")
            builder.ExportOutput(rgbd.depth_image_32F_output_port(),
                                 f"{model_name}_depth_image")
            builder.ExportOutput(rgbd.label_image_output_port(),
                                 f"{model_name}_label_image")
            
            if also_add_point_clouds:
                to_point_cloud = builder.AddSystem(
                        DepthImageToPointCloud(camera_info=rgbd.depth_camera_info(),
                            fields = BaseField.kXYZs 
                            | BaseField.kRGBs))
                builder.Connect(rgbd.depth_image_32F_output_port(),
                                to_point_cloud.depth_image_input_port())
                builder.Connect(rgbd.color_image_output_port(), 
                                to_point_cloud.color_image_input_port())

                class ExtractBodyPose(LeafSystem):

                    def __init__(self, body_index):
                        LeafSystem.__init__(self)
                        self.body_index = body_index
                        self.DeclareAbstractInputPort(
                                "poses",
                                plant.get_body_poses_output_port().Allocate())
                        self.DeclareAbstractOutputPort(
                                "pose",
                                lambda: AbstractValue.Make(RigidTransform()),
                                self.CalcOutput)
                        
                    def CalcOutput(self, context, output):
                        poses = self.EvalAbstractInput(context, 0).get_value()
                        pose = poses[int(self.body_index)]
                        output.get_mutable_value().set(pose.rotation(),
                                                       pose.translation())

                camera_pose = builder.AddSystem(ExtractBodyPose(body_index))
                builder.Connect(plant.get_body_poses_output_port(),
                                camera_pose.get_input_port())
                builder.Connect(camera_pose.get_output_port(),
                                to_point_cloud.GetInputPort("camera_pose"))
                builder.ExportOutput(to_point_cloud.point_cloud_output_port(),
                                     f"{model_name}_point_cloud")

def AddPandaArmHand(plant, q0 = [0.0, 0.1, 0, -1.2, 0, 1.6, 0], 
        X_WB =  RigidTransform(), welded = False):
    """ Adds a franka panda arm and hand to the multibody plant and welds its base
    to the world frame

    plant: the multibody plant to add the panda to
    q0: the initial joint positions 
    X_WB: the desired transformation between the world frame (W) and the base link
    of the panda (B)
    welded: True => the finger joints of the panda will be welded in the open 
    position
    """
    if welded:
        urdf_file = pydrake.common.FindResourceOrThrow(
                "./models/panda_arm_hand_welded.urdf")
    else:
        urdf_file = pydrake.common.FindResourceOrThrow(
                "./models/panda_arm_hand.urdf")

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
        gripper = parser.AddModelFromFile(FindResource("models/welded_panda_hand.urdf"))
    else:
        gripper = parser.AddModelFromFile(
                FindResource("models/panda_hand_fixed_collisions.urdf"))
        #gripper = parser.AddModelFromFile(pydrake.common.FindResourceOrThrow("drake/manipulation/models/franka_description/urdf/hand.urdf"))

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
        inertia = pydrake.multibody.tree.UnitInertia.SolidBox(
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





