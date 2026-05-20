import copy
import math

import mujoco
import numpy as np
import numpy.typing as npt

from robosuite import macros
from robosuite.controllers.dynamics.robot_dynamics_model import RoboDynamicsModel
from robosuite.controllers.osc import OperationalSpaceController
from robosuite.utils.signal_processing_utils import BackwardEulerDiff, LowPassFilter
import robosuite.utils.transform_utils as T
from robosuite.utils.control_utils import *

# Supported impedance modes
IMPEDANCE_MODES = {"fixed", "variable", "variable_kp"}

# TODO: Maybe better naming scheme to differentiate between input / output min / max and pos/ori limits, etc.

def generate_random_vector_w_specified_magnitude(shape : tuple, magnitude : float, np_random : np.random.Generator) -> npt.NDArray[np.float64]:   
    out_vec = np_random.random(*shape)
    out_vec /= np.linalg.norm(out_vec)
    out_vec *= magnitude
    return out_vec

class OSCWithNominalModel(OperationalSpaceController):
    """
    Extension of OSC controller where the controller takes in a nominal model of the robot that is not affected by 
    the changes in the simulation, so that one can simulate misalignment between the assumed model and the actual
    robot. This is useful for training policies that can compensate for model errors.

    Args:
        sim (MjSim): Simulator instance this controller will pull robot state updates from

        eef_name (str): Name of controlled robot arm's end effector (from robot XML)

        joint_indexes (dict): Each key contains sim reference indexes to relevant robot joint information, namely:

            :`'joints'`: list of indexes to relevant robot joints
            :`'qpos'`: list of indexes to relevant robot joint positions
            :`'qvel'`: list of indexes to relevant robot joint velocities

        actuator_range (2-tuple of array of float): 2-Tuple (low, high) representing the robot joint actuator range

        input_max (float or Iterable of float): Maximum above which an inputted action will be clipped. Can be either be
            a scalar (same value for all action dimensions), or a list (specific values for each dimension). If the
            latter, dimension should be the same as the control dimension for this controller

        input_min (float or Iterable of float): Minimum below which an inputted action will be clipped. Can be either be
            a scalar (same value for all action dimensions), or a list (specific values for each dimension). If the
            latter, dimension should be the same as the control dimension for this controller

        output_max (float or Iterable of float): Maximum which defines upper end of scaling range when scaling an input
            action. Can be either be a scalar (same value for all action dimensions), or a list (specific values for
            each dimension). If the latter, dimension should be the same as the control dimension for this controller

        output_min (float or Iterable of float): Minimum which defines upper end of scaling range when scaling an input
            action. Can be either be a scalar (same value for all action dimensions), or a list (specific values for
            each dimension). If the latter, dimension should be the same as the control dimension for this controller

        kp (float or Iterable of float): positional gain for determining desired torques based upon the pos / ori error.
            Can be either be a scalar (same value for all action dims), or a list (specific values for each dim)

        damping_ratio (float or Iterable of float): used in conjunction with kp to determine the velocity gain for
            determining desired torques based upon the joint pos errors. Can be either be a scalar (same value for all
            action dims), or a list (specific values for each dim)

        impedance_mode (str): Impedance mode with which to run this controller. Options are {"fixed", "variable",
            "variable_kp"}. If "fixed", the controller will have fixed kp and damping_ratio values as specified by the
            @kp and @damping_ratio arguments. If "variable", both kp and damping_ratio will now be part of the
            controller action space, resulting in a total action space of (6 or 3) + 6 * 2. If "variable_kp", only kp
            will become variable, with damping_ratio fixed at 1 (critically damped). The resulting action space will
            then be (6 or 3) + 6.

        kp_limits (2-list of float or 2-list of Iterable of floats): Only applicable if @impedance_mode is set to either
            "variable" or "variable_kp". This sets the corresponding min / max ranges of the controller action space
            for the varying kp values. Can be either be a 2-list (same min / max for all kp action dims), or a 2-list
            of list (specific min / max for each kp dim)

        damping_ratio_limits (2-list of float or 2-list of Iterable of floats): Only applicable if @impedance_mode is
            set to "variable". This sets the corresponding min / max ranges of the controller action space for the
            varying damping_ratio values. Can be either be a 2-list (same min / max for all damping_ratio action dims),
            or a 2-list of list (specific min / max for each damping_ratio dim)

        policy_freq (int): Frequency at which actions from the robot policy are fed into this controller

        position_limits (2-list of float or 2-list of Iterable of floats): Limits (m) below and above which the
            magnitude of a calculated goal eef position will be clipped. Can be either be a 2-list (same min/max value
            for all cartesian dims), or a 2-list of list (specific min/max values for each dim)

        orientation_limits (2-list of float or 2-list of Iterable of floats): Limits (rad) below and above which the
            magnitude of a calculated goal eef orientation will be clipped. Can be either be a 2-list
            (same min/max value for all joint dims), or a 2-list of list (specific min/mx values for each dim)

        interpolator_pos (Interpolator): Interpolator object to be used for interpolating from the current position to
            the goal position during each timestep between inputted actions

        interpolator_ori (Interpolator): Interpolator object to be used for interpolating from the current orientation
            to the goal orientation during each timestep between inputted actions

        control_ori (bool): Whether inputted actions will control both pos and ori or exclusively pos

        control_delta (bool): Whether to control the robot using delta or absolute commands (where absolute commands
            are taken in the world coordinate frame)

        uncouple_pos_ori (bool): Whether to decouple torques meant to control pos and torques meant to control ori

        **kwargs: Does nothing; placeholder to "sink" any additional arguments so that instantiating this controller
            via an argument dict that has additional extraneous arguments won't raise an error

    Raises:
        AssertionError: [Invalid impedance mode]
    """

    def __init__(
        self,
        sim,
        eef_name,
        joint_indexes,
        actuator_range,
        input_max=1,
        input_min=-1,
        output_max=(0.05, 0.05, 0.05, 0.5, 0.5, 0.5),
        output_min=(-0.05, -0.05, -0.05, -0.5, -0.5, -0.5),
        kp=150,
        damping_ratio=1,
        impedance_mode="fixed",
        kp_limits=(0, 300),
        damping_ratio_limits=(0, 100),
        policy_freq=20,
        position_limits=None,
        orientation_limits=None,
        interpolator_pos=None,
        interpolator_ori=None,
        control_ori=True,
        control_delta=True,
        uncouple_pos_ori=True,
        stiffness_in_tool_frame=True,
        nominal_model_urdf_fp=None,
        armature=None,
        enable_disturbance=False,
        custom_disturbance_torque_fn=None,
        max_disturbance_torque_mag=0.0,
        delay_control=False,
        simulate_stribeck_friction=True,
        perfect_gravity_compensation=False,
        perfect_inertial_parameters=False,
        np_random = None,
        **kwargs,  # does nothing; used so no error raised when dict is passed with extra terms used previously
    ):
        # assert nominal_model_urdf_fp is not None, "Must provide a nominal model URDF filepath for OSCWithNominalModel"
        self.nominal_robot_model = RoboDynamicsModel(nominal_model_urdf_fp, armature=armature, sim=sim)
        
        self.enable_disturbance = enable_disturbance
        self.custom_disturbance_torques_fn = custom_disturbance_torque_fn
        if self.custom_disturbance_torques_fn is None and self.enable_disturbance:
            self.max_disturbance_torque = self.nums2array(max_disturbance_torque_mag, self.nominal_robot_model.n_dof)
            self.min_disturbance_torque = -self.max_disturbance_torque
        self.perfect_inertial_parameters = perfect_inertial_parameters
    
        super().__init__(
            sim,
            eef_name=eef_name,
            joint_indexes=joint_indexes,
            actuator_range=actuator_range,
            input_max=input_max,
            input_min=input_min,
            output_max=output_max,
            output_min=output_min,
            kp=kp,
            damping_ratio=damping_ratio,
            impedance_mode=impedance_mode,
            kp_limits=kp_limits,
            damping_ratio_limits=damping_ratio_limits,
            policy_freq=policy_freq,
            position_limits=position_limits,
            orientation_limits=orientation_limits,
            interpolator_pos=interpolator_pos,
            interpolator_ori=interpolator_ori,
            control_ori=control_ori,
            control_delta=control_delta,
            uncouple_pos_ori=uncouple_pos_ori,
            stiffness_in_tool_frame=stiffness_in_tool_frame,
            np_random=np_random,
            **kwargs,
        )

        if self.enable_disturbance and self.custom_disturbance_torques_fn is None:
            self.disturbance_joint_torque = np.empty(self.nominal_robot_model.n_dof, np.float64)
            self.calculate_disturbance_joint_torque()
        else:
            self.disturbance_joint_torque = None
            
        self.torque_filter = LowPassFilter(300.0, self.dt)

        self.joint_pos_filter = LowPassFilter(300.0, self.dt)

        self.joint_vel_eul_diff = BackwardEulerDiff(self.dt)
        self.joint_vel_filter = LowPassFilter(300.0, self.dt)

        self.joint_accel = None
        self.joint_accel_eul_diff = BackwardEulerDiff(self.dt)
        self.joint_accel_filter = LowPassFilter(1.0, self.dt)
        
        self.delay_control = delay_control
        self.torques_buffer = None
        
        self.simulate_stribeck_friction = simulate_stribeck_friction
        self.perfect_gravity_compensation = perfect_gravity_compensation

    def update(self, force=False):
        """
        Updates the state of the robot arm, including end effector pose / orientation / velocity, joint pos/vel,
        jacobian, and mass matrix. By default, since this is a non-negligible computation, multiple redundant calls
        will be ignored via the self.new_update attribute flag. However, if the @force flag is set, the update will
        occur regardless of that state of self.new_update. This base class method of @run_controller resets the
        self.new_update flag

        Args:
            force (bool): Whether to force an update to occur or not
        """

        # Only run update if self.new_update or force flag is set
        if self.new_update or force:
            # TODO: remove superclass call and replace with custom update
            super(OperationalSpaceController, self).update()
            
            self.joint_accel = np.array(self.sim.data.qacc[self.qvel_index])

            self.nominal_robot_model.update_model(self.joint_pos, self.joint_vel, self.joint_accel)

            self.J_full = self.nominal_robot_model.J_full
            self.J_pos = self.nominal_robot_model.J_pos
            self.J_ori = self.nominal_robot_model.J_ori

            self.ee_pos = self.nominal_robot_model.ee_pos
            self.ee_ori_mat = self.nominal_robot_model.ee_ori
            self.ee_pos_vel = self.J_pos @ self.joint_vel
            self.ee_ori_vel = self.J_ori @ self.joint_vel

            if self.perfect_inertial_parameters:
                mass_matrix = np.ndarray(shape=(self.sim.model.nv, self.sim.model.nv), dtype=np.float64, order="C")
                mujoco.mj_fullM(self.sim.model._model, mass_matrix, self.sim.data.qM)
                mass_matrix = np.reshape(mass_matrix, (len(self.sim.data.qvel), len(self.sim.data.qvel)))
                self.mass_matrix = mass_matrix[self.qvel_index, :][:, self.qvel_index]
                self.mass_matrix_inv = np.linalg.inv(self.mass_matrix)
            else:
                self.mass_matrix = self.nominal_robot_model.mass_matrix
                self.mass_matrix_inv = self.nominal_robot_model.mass_matrix_inv

            # Clear self.new_update
            self.new_update = False

    def run_controller(self):
        """
        Calculates the torques required to reach the desired setpoint.

        Executes Operational Space Control (OSC) -- either position only or position and orientation.

        A detailed overview of derivation of OSC equations can be seen at:
        http://khatib.stanford.edu/publications/pdfs/Khatib_1987_RA.pdf

        Returns:
             np.array: Command torques
        """
        # Update state
        self.update()

        desired_pos = None
        # Only linear interpolator is currently supported
        if self.interpolator_pos is not None:
            # Linear case
            if self.interpolator_pos.order == 1:
                desired_pos = self.interpolator_pos.get_interpolated_goal()
            else:
                # Nonlinear case not currently supported
                pass
        else:
            desired_pos = np.array(self.goal_pos)

        if self.interpolator_ori is not None:
            # relative orientation based on difference between current ori and ref
            self.relative_ori = orientation_error(self.ee_ori_mat, self.ori_ref)

            ori_error = self.interpolator_ori.get_interpolated_goal()
        else:
            desired_ori = np.array(self.goal_ori)
            ori_error = orientation_error(desired_ori, self.ee_ori_mat)

        # Compute desired force and torque based on errors
        position_error = desired_pos - self.ee_pos
        vel_pos_error = -self.ee_pos_vel

        vel_ori_error = -self.ee_ori_vel

        if self.stiffness_in_tool_frame:
            # Transform the position error to be in the tool frame
            position_error = self.ee_ori_mat @ position_error
            vel_pos_error = self.ee_ori_mat @ vel_pos_error
            ori_error = self.ee_ori_mat @ ori_error
            vel_ori_error = self.ee_ori_mat @ vel_ori_error

        # F_r = kp * pos_err + kd * vel_err
        desired_force = np.multiply(np.array(position_error), np.array(self.kp[0:3])) + np.multiply(
            vel_pos_error, self.kd[0:3]
        )

        # Tau_r = kp * ori_err + kd * vel_err
        desired_torque = np.multiply(np.array(ori_error), np.array(self.kp[3:6])) + np.multiply(
            vel_ori_error, self.kd[3:6]
        )
        
        if self.perfect_inertial_parameters:
            lambda_full, lambda_pos, lambda_ori, nullspace_matrix = opspace_matrices(
                self.mass_matrix, self.J_full, self.J_pos, self.J_ori
            )
        else:
            lambda_full = self.nominal_robot_model.lambda_full
            lambda_pos = self.nominal_robot_model.lambda_pos
            lambda_ori = self.nominal_robot_model.lambda_ori
            nullspace_matrix = self.nominal_robot_model.nullspace_matrix

        # Decouples desired positional control from orientation control
        if self.uncoupling:
            decoupled_force = np.dot(lambda_pos, desired_force)
            decoupled_torque = np.dot(lambda_ori, desired_torque)
            decoupled_wrench = np.concatenate([decoupled_force, decoupled_torque])
        else:
            desired_wrench = np.concatenate([desired_force, desired_torque])
            decoupled_wrench = np.dot(lambda_full, desired_wrench)

        if self.stiffness_in_tool_frame:
            # Transform the desired force and torque to be in the world frame
            decoupled_wrench[:3] = self.ee_ori_mat.T @ decoupled_wrench[:3]
            decoupled_wrench[3:] = self.ee_ori_mat.T @ decoupled_wrench[3:]
        
        # Gamma (without null torques) = J^T * F + gravity compensations
        computed_torques = np.dot(self.J_full.T, decoupled_wrench)

        # Calculate and add nullspace torques (nullspace_matrix^T * Gamma_null) to final torques
        # Note: Gamma_null = desired nullspace pose torques, assumed to be positional joint control relative
        #                     to the initial joint positions
        computed_torques += nullspace_torques(
            self.mass_matrix, nullspace_matrix, self.initial_joint, self.joint_pos, self.joint_vel
        )

        computed_torques += self.torque_compensation
        
        filtered_torques = self.torque_filter(computed_torques)

        # Apply control delay
        if self.delay_control:
            if self.torques_buffer is None:
                # On first step, apply gravity compensation only
                self.torques_buffer = np.empty_like(filtered_torques)
                self.torques = self.torque_compensation.copy()
            else:
                # Apply what was computed in the previous step
                self.torques[:] = self.torques_buffer[:]

            # Store current result for next step
            self.torques_buffer[:] = filtered_torques[:]
        else:
            self.torques = filtered_torques
            
        if self.enable_disturbance:
            if self.custom_disturbance_torques_fn is not None:
                self.disturbance_joint_torque = self.custom_disturbance_torques_fn(self.joint_pos, self.joint_vel, self.joint_accel)
            self.torques += self.disturbance_joint_torque
            
        if self.simulate_stribeck_friction:
            self.torques -= self.calculate_stribeck_friction()

        super(OperationalSpaceController, self).run_controller()
        return self.torques

    def calculate_disturbance_joint_torque(self):
        self.disturbance_joint_torque = self.np_random.uniform(self.min_disturbance_torque, self.max_disturbance_torque)
        
    def calculate_stribeck_friction(self):
        v = self.joint_vel
        Fc = self.sim.model.dof_frictionloss[self.joint_index]       # Coulomb friction already in MuJoCo
        Fs = self.sim.model.jnt_user[self.joint_index, 0] * Fc       # Static friction (expressed as multiple of Coulomb friction)
        vs = self.sim.model.jnt_user[self.joint_index, 1]            # Stribeck velocity
        alpha = self.sim.model.jnt_user[self.joint_index, 2]         # Stribeck exponent

        # Extra over Coulomb (MuJoCo already handles Fc + viscous)
        stribeck_mag = (Fs - Fc) * np.exp(-(np.abs(v) / np.maximum(vs, 1e-9)) ** alpha)

        # thresholds
        v_eps = 1e-4   # small velocity deadband
        tau_eps = 1e-6 # small torque margin

        tau_f = np.zeros_like(v)

        # 1) STICK: near-zero velocity and not exceeding breakaway
        stick_mask = (np.abs(v) < v_eps) & (np.abs(self.torques) <= (Fs - tau_eps))

        # In stick, subtract the command up to the EXTRA margin (Fs - Fc)
        extra_max = np.maximum(Fs - Fc, 0)
        tau_f[stick_mask] = np.clip(self.torques[stick_mask], -extra_max[stick_mask], extra_max[stick_mask])

        # 2) SLIP: otherwise, subtract along sign of velocity (so net opposes motion)
        slip_mask = ~stick_mask
        tau_f[slip_mask] = np.sign(v[slip_mask]) * stribeck_mag[slip_mask]

        # IMPORTANT: caller does: self.torques -= tau_f
        return tau_f
        
    @property
    def name(self):
        return "OSC_NOMINAL_MODEL_" + self.name_suffix
    
    @property
    def torque_compensation(self):
        """
        Gravity compensation for this robot arm

        Returns:
            np.array: torques
        """
        if self.perfect_gravity_compensation:
            return self.sim.data.qfrc_bias[self.joint_index]
        else:
            self.nominal_robot_model.compute_gravity_torque(self.joint_pos)
            self.nominal_robot_model.compute_coriolis_matrix(self.joint_pos, self.joint_vel)
            return self.nominal_robot_model.torque_gravity + self.nominal_robot_model.coriolis_matrix @ self.joint_vel
    
    def update_base_pose(self, base_pos, base_ori):
        """
        Optional function to implement in subclass controllers that will take in @base_pos and @base_ori and update
        internal configuration to account for changes in the respective states. Useful for controllers e.g. IK, which
        is based on pybullet and requires knowledge of simulator state deviations between pybullet and mujoco

        Args:
            base_pos (3-tuple): x,y,z position of robot base in mujoco world coordinates
            base_ori (4-tuple): x,y,z,w orientation or robot base in mujoco world coordinates
        """
        self.nominal_robot_model.update_base_pose(base_pos, base_ori)
