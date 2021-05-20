from .RRT import RRT
import pydrake.all
from PandaInverseKinematics import PandaInverseKinematics


class RRTPandaPlanner:

    def __init__(self, plant, scene_graph, 
                 plant_context, scene_graph_context, panda 
                 start_pose, goal_pose, avoid_names,
                 p_tol = 0.01, theta_tol = 0.01):
        """
        A wrapper class around RRT for motion planning with the franka-panda

        plant: the mulibody plant 
        scene_graph: the scene graph
        plant_context: the plant context
        scene_graph_context: the scene graph context
        panda; the model instance of the panda arm
        start_pose: the desired starting pose of the panda end effector
        goal_pose: the  desired ending pose of the panda end effector
        p_tol: the tolerance in the translation of the end effector (meters)
        theta_tol: the tolerance in the rotation of the end effector (in radians)
        """

        self.plant = plant
        self.scene_graph = scene_graph
        self.plant_context = plant_context
        self.scene_graph_context = scene_graph_context
        self.panda = panda
        self.start_pose = start_pose
        self.goal_pose = goal_pose
        self.avoid_names = avoid_names
        self.p_tol = p_tol
        self.theta_tol = theta_tol

        self.ik = PandaInverseKinematics(self.plant, self.plant_context,
                                         self.panda, self.avoid_names)


    def is_colliding(self, q):
        """
        returns True if the configuration q results in a collision,
        False otherwise
        """
        pass


    def CreateRRT(self):
        """
        Creats the RRT connecting q_start to q_goal and returns the 
        intermediate joint configurations in a np.array
        """
        pass


