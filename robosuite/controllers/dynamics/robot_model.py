import tempfile
from copy import deepcopy
import numpy as np
from urdf_parser_py.urdf import URDF 
import pinocchio

from robosuite.utils.control_utils import inverse_cholesky
import robosuite.utils.transform_utils as T

class RobotModel:
    def __init__(self, urdf_str, ee_link):
        self.parsed_urdf_model = URDF.from_xml_string(urdf_str) # parsed urdf model for convenience

        with tempfile.NamedTemporaryFile(mode='wb', delete=True, suffix=".urdf") as temp_file:
            temp_file.write(urdf_str)
            temp_file.flush()

            self.model, _, _ = pinocchio.buildModelsFromUrdf(
                filename=temp_file.name,
            )
        
        # self.model.gravity.setZero()
        self.data = self.model.createData()
        self.ee_link_frame_id = self.model.getFrameId(ee_link)

        self.fetch_joint_friction_and_damping()

    def update_model(self, q, qd, qdd):
        pinocchio.forwardKinematics(self.model, self.data, q)
        pinocchio.updateFramePlacements(self.model, self.data)

    def compute_eef_pose(self, q, pos, ori):
        eef_se3 = self.data.oMf[self.ee_link_frame_id]
        pos[:] = eef_se3.translation[:]
        ori[:] = eef_se3.rotation[:]

    def compute_eef_jacobian(self, q, J_full):
        J_full[:] = pinocchio.computeFrameJacobian(
            self.model, self.data, q, self.ee_link_frame_id, 
            pinocchio.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )[:]

        J_full[:] = np.vstack((J_full[3:], J_full[:3]))[:]

    def compute_mass_matrix(self, q, mass_matrix):
        mass_matrix[:] = 0.0
        mass_matrix[:] = pinocchio.crba(self.model, self.data, q)[:]

    def compute_coriolis_matrix(self, q, qd, coriolis_matrix):
        coriolis_matrix[:] = pinocchio.computeCoriolisMatrix(self.model, self.data, q, qd)[:]

    def compute_gravity_torque(self, q, torque_gravity):
        torque_gravity[:] = pinocchio.computeGeneralizedGravity(self.model, self.data, q)[:]

    def compute_operational_space_matrices(
        self, uncoupling, mass_matrix_inv, J_full, J_pos, J_ori, lambda_full, lambda_pos, lambda_ori 
    ):
        
        # J M^-1 J^T
        lambda_full_inv = np.dot(np.dot(J_full, mass_matrix_inv), J_full.transpose())
        lambda_full[:] = inverse_cholesky(lambda_full_inv)
        
        if not uncoupling:
            return
        
        # Jx M^-1 Jx^T
        lambda_pos_inv = np.dot(np.dot(J_pos, mass_matrix_inv), J_pos.transpose())
        lambda_pos[:] = inverse_cholesky(lambda_pos_inv)

        # Jr M^-1 Jr^T
        lambda_ori_inv = np.dot(np.dot(J_ori, mass_matrix_inv), J_ori.transpose())
        lambda_ori[:] = inverse_cholesky(lambda_ori_inv)

    
    @property
    def effort_limits(self):
        lower_limit =  np.array(
            [
                -joint.limit.effort
                for joint in self.parsed_urdf_model.joints
                if joint.joint_type != "fixed"
            ]
        )

        upper_limit =  np.array(
            [
                joint.limit.effort
                for joint in self.parsed_urdf_model.joints
                if joint.joint_type != "fixed"
            ]
        )

        return lower_limit, upper_limit
    
    @property 
    def n_dof(self):
        return self.model.nq
    
    def fetch_joint_friction_and_damping(self):
        joint_frictions = []
        joint_dampings = []
        for joint in self.parsed_urdf_model.joints:
            if joint.joint_type == "fixed":
                continue

            joint_frictions.append(joint.dynamics.friction)
            joint_dampings.append(joint.dynamics.damping)

        self.joint_frictions = np.array(joint_frictions, dtype=np.float64)
        self.joint_dampings = np.array(joint_dampings, dtype=np.float64)
    
    def compute_friction_and_damping_compensation(
            self, tau, qd, torques_friction, 
            stiction_positive=np.array([0.0, 0.0, 0.0, 0.0, 0.0, .85, 0.3]), #.15
            stiction_negative=np.array([0.0, 0.0, 0.0, 0.0, 0., .85, 0.3]) #.3
        ):
        tau_sgn = np.sign(tau)
        qd_sgn = np.sign(qd)

        coulomb_friction_torque = np.multiply(qd_sgn, self.joint_frictions)
        damping_torque = np.multiply(qd, self.joint_dampings)
        
        # not_low_speed_condition = ~np.isclose(qd, 0.0, rtol=1e-20)
        # zero_cmd_torque_condition = np.isclose(tau, 0.0, rtol=1e-5)
        # combined_condition = not_low_speed_condition & zero_cmd_torque_condition

        # joint_indices_not_stiction_compensated = np.where(
        #     combined_condition
        # )
        stiction_torque = np.zeros_like(coulomb_friction_torque)
        stiction_torque[tau > 0.0] = stiction_positive[tau > 0.0]
        stiction_torque[tau <= 0.0] = stiction_negative[tau <= 0.0]
        stiction_torque = np.multiply(tau_sgn, stiction_torque)
        # stiction_torque[joint_indices_not_stiction_compensated] = 0.0

        # stiction_torque[np.abs(qd) > 1e-1] = 0.0
        print(qd)
        torques_friction[:] = 0.0
        np.add(torques_friction, coulomb_friction_torque, out=torques_friction)
        np.add(torques_friction, damping_torque, out=torques_friction)
        np.add(torques_friction, stiction_torque, out=torques_friction)

        # torques_friction[np.abs(tau) < 1e-2] = 0.0

    def compute_inertia_weight_jac_psuedo_inv(self, mass_matrix_inv, J_full, lambda_full, J_bar):
        np.dot(mass_matrix_inv, J_full.T, out=J_bar)
        np.dot(J_bar,lambda_full, out=J_bar)

    def compute_nullspace_matrix(self, J_full, Jbar, nullspace_matrix):
        eye_temp = np.eye(self.n_dof)
        np.dot(Jbar, J_full, out=nullspace_matrix)
        np.subtract(eye_temp, nullspace_matrix, out=nullspace_matrix)
