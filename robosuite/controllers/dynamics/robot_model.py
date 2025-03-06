import tempfile
from copy import deepcopy
import numpy as np
from urdf_parser_py.urdf import URDF 
import pinocchio # type: ignore

from robosuite.utils.control_utils import inverse_cholesky
import robosuite.utils.transform_utils as T

class RoboDynamicsModel:
    def __init__(self, urdf_fp, ee_link):
        self.parsed_urdf_model = URDF.from_xml_file(urdf_fp) # parsed urdf model for convenience
        
        self.model, _, _ = pinocchio.buildModelsFromUrdf(
            filename=urdf_fp,
        )

        self.data = self.model.createData()
        self.ee_link_frame_id = self.model.getFrameId(ee_link)

        self.fetch_joint_friction_and_damping()

        self.ee_pos = np.empty(3, dtype=np.float64)
        self.ee_ori = np.empty((3,3), dtype=np.float64)
        self.J_full = np.empty((6, self.n_dof), dtype=np.float64)
        self.J_pos = np.empty((3, self.n_dof), dtype=np.float64)
        self.J_ori = np.empty((3, self.n_dof), dtype=np.float64)
        self.mass_matrix = np.empty((self.n_dof, self.n_dof), dtype=np.float64)
        self.mass_matrix_inv = np.empty((self.n_dof, self.n_dof), dtype=np.float64)
        self.coriolis_matrix = np.empty((self.n_dof, self.n_dof), dtype=np.float64)
        self.torque_gravity = np.empty(self.n_dof, dtype=np.float64)
        self.J_bar = np.empty((self.n_dof, 6), dtype=np.float64)
        self.lambda_full = np.empty((6, 6), dtype=np.float64)
        self.lambda_full_inv = np.empty((6, 6), dtype=np.float64)
        self.lambda_pos = np.empty((3, 3), dtype=np.float64) 
        self.lambda_pos_inv = np.empty((3, 3), dtype=np.float64)   
        self.lambda_ori = np.empty((3, 3), dtype=np.float64)
        self.lambda_ori_inv = np.empty((3, 3), dtype=np.float64)   
        self.nullspace_matrix = np.empty((self.n_dof, self.n_dof), dtype=np.float64)
        self.torques_friction = np.empty(self.n_dof, dtype=np.float64)  

    def update_model(self, q, qd, qdd):
        pinocchio.forwardKinematics(self.model, self.data, q)
        pinocchio.updateFramePlacements(self.model, self.data)

    def compute_eef_pose(self, q):
        eef_se3 = self.data.oMf[self.ee_link_frame_id]
        self.ee_pos[:] = eef_se3.translation[:]
        self.ee_ori[:] = eef_se3.rotation[:]

    def compute_eef_jacobian(self, q):
        self.J_full[:] = pinocchio.computeFrameJacobian(
            self.model, self.data, q, self.ee_link_frame_id, 
            pinocchio.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )[:]

        self.J_pos[:] = self.J_full[:3, :]
        self.J_ori = self.J_full[3:, :]

    def compute_mass_matrix(self, q):
        self.mass_matrix[:] = 0.0
        self.mass_matrix[:] = pinocchio.crba(self.model, self.data, q)[:]
        self.mass_matrix_inv[:] = inverse_cholesky(self.mass_matrix)

    def compute_coriolis_matrix(self, q, qd):
        self.coriolis_matrix[:] = pinocchio.computeCoriolisMatrix(self.model, self.data, q, qd)[:]

    def compute_gravity_torque(self, q):
        self.torque_gravity[:] = pinocchio.computeGeneralizedGravity(self.model, self.data, q)[:]

    def compute_operational_space_matrices(self):
        
        # J M^-1 J^T
        self.lambda_full_inv[:] = np.dot(np.dot(self.J_full, self.mass_matrix_inv), self.J_full.transpose())
        self.lambda_full[:] = inverse_cholesky(self.lambda_full_inv)
        
        # Jx M^-1 Jx^T
        self.lambda_pos_inv[:] = np.dot(np.dot(self.J_pos, self.mass_matrix_inv), self.J_pos.transpose())
        self.lambda_pos[:] = inverse_cholesky(self.lambda_pos_inv)

        # Jr M^-1 Jr^T
        self.lambda_ori_inv[:] = np.dot(np.dot(self.J_ori, self.mass_matrix_inv), self.J_ori.transpose())
        self.lambda_ori[:] = inverse_cholesky(self.lambda_ori_inv)

    
    def compute_inertia_weight_jac_psuedo_inv(self):
        np.dot(self.mass_matrix_inv, self.J_full.T, out=self.J_bar)
        np.dot(self.J_bar,self.lambda_full, out=self.J_bar)

    def compute_nullspace_matrix(self):
        eye_temp = np.eye(self.n_dof)
        np.dot(self.J_bar, self.J_full, out=self.nullspace_matrix)
        np.subtract(eye_temp, self.nullspace_matrix, out=self.nullspace_matrix)


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
