"""
Controller system for panda hand.
reference: https://github.com/RussTedrake/drake/blob/master/manipulation/schunk_wsg/schunk_wsg_position_controller.cc
"""

import pydrake.all
import numpy as np
from pydrake.systems.framework import BasicVector


def MakeMultibodyStateToPandaHandStateSystem():
    D = np.array([[-1, -1, 0, 0],
                 [0, 0, -1, -1]])
    return pydrake.systems.primitives.MatrixGain(D)
    

class PandaHandPdController(pydrake.systems.framework.LeafSystem):

    #TODO(ben): make sure these controller values are realistic
    def __init__(self, kp_command = 200., kd_command = 5., 
                 kp_constraint = 2000., kd_constraint = 5.,
                 default_force_limit = 40.):
        pydrake.systems.framework.LeafSystem.__init__(self)

        self.num_joints = 2
        
        self.kp_command = kp_command
        self.kd_command = kd_command
        self.kp_constraint = kp_constraint
        self.kd_constraint = kd_constraint
        self.default_force_limit = default_force_limit

        self.desired_state_input_port = self.DeclareVectorInputPort(
                "desired_state", 
                BasicVector(2))
        self.force_limit_input_port = self.DeclareVectorInputPort(
                "force_limit", 
                BasicVector(1))
        self.state_input_port = self.DeclareVectorInputPort(
                "state", 
                BasicVector(2 * self.num_joints))

        self.generalized_force_output_port = self.DeclareVectorOutputPort(
                "generalized_force",
                BasicVector(self.num_joints),
                self.CalcGeneralizedForceOutput)
        self.grip_force_output_port = self.DeclareVectorOutputPort(
                "grip_force",
                BasicVector(1),
                self.CalcGripForceOutput)

        self.set_name("panda_hand_controller")


    def get_desired_state_input_port(self):
        return get_input_port(self.desired_state_input_port)
        
    def get_force_limit_input_port(self):
        return self.force_limit_input_port

    def get_state_input_port(self):
        return self.state_input_port

    def get_generalized_force_output_port(self):
        return self.generalized_force_output_port

    def get_grip_force_output_port(self):
        return self.grip_force_output_port

    def CalcGeneralizedForce(self, context):
        desired_state = self.desired_state_input_port.Eval(context)
        if (self.force_limit_input_port.HasValue(context)):
            force_limit = self.force_limit_input_port.Eval(context)[0]
        else:
            force_limit = self.default_force_limit

        if (force_limit <= 0):
            raise Exception("Force limit must be greater than 0")

        state = self.state_input_port.Eval(context)

        f0_plus_f1 = -self.kp_constraint * (state[0] + state[1]) - self.kd_constraint * (state[2] + state[3])

        neg_f0_plus_f1 = self.kp_command * (desired_state[0] + state[0] - state[1]) + self.kd_command * (desired_state[1] + state[2] - state[3])

        if (neg_f0_plus_f1 > force_limit):
            neg_f0_plus_f1 = force_limit
        if (neg_f0_plus_f1 < -force_limit):
            neg_f0_plut_f1 = -force_limit

        return np.array([0.5*f0_plus_f1-0.5*neg_f0_plus_f1, 0.5*f0_plus_f1 + 0.5*neg_f0_plus_f1np])


    def CalcGeneralizedForceOutput(self, context, output_vector):
        output_vector.SetFromVector(self.CalcGeneralizedForce(context))

    def CalcGripForceOutput(self, context, output_vector):
        force = self.CalcGeneralizedForce(context)
        output_vector.SetAtIndex(0, np.abs(force[0] - force[1]))

class PandaHandPositionController(pydrake.systems.framework.Diagram):

    def __init__(self, 
            time_step = 0.05,
            kp_command = 200.,
            kd_command = 5.,
            kp_constraint = 2000.,
            kd_constraint = 5.,
            default_force_limit = 40.):
        pydrake.systems.framework.Diagram.__init__(self)

        self.time_step = time_step
        self.kp_command = kp_command
        self.kd_command = kd_command
        self.kp_constraint = kp_constraint
        self.kd_constraint = kd_constraint
        self.default_force_limit = default_force_limit

        builder = pydrake.systems.framework.DiagramBuilder()
        self.pd_controller = builder.AddSystem(
                PandaHandPdController(
                    kp_command,
                    kd_command,
                    kp_constraint,
                    kd_constraint,
                    default_force_limit))
        self.state_interpolator = builder.AddSystem(
                pydrake.systems.primitives.StateInterpolatorWithDiscreteDerivative(
                    1, time_step, suppress_initial_transient = True))

        self.desired_position_input_port = builder.ExportInput(
                self.state_interpolator.get_input_port(), "desired_position")
        self.force_limit_input_port = builder.ExportInput(
                self.pd_controller.get_force_limit_input_port(), "force_limit")
        self.state_input_port = builder.ExportInput(
                self.pd_controller.get_state_input_port(), "state")

        self.generalized_force_output_port = builder.ExportOutput(
                self.pd_controller.get_generalized_force_output_port(), "generalized_force")
        self.grip_force_output_port = builder.ExportOutput(
                self.pd_controller.get_grip_force_output_port(), "grip_force")

        builder.BuildInto(self)




