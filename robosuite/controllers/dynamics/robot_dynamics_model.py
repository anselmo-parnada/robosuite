import tempfile
from copy import deepcopy
import numpy as np
from urdf_parser_py.urdf import URDF 
import pinocchio # type: ignore

from robosuite.utils.control_utils import inverse_cholesky
import robosuite.utils.transform_utils as T
import xml.etree.ElementTree as ET

IDENT_QUAT = np.array([1., 0., 0., 0.], np.float64)

def extract_robot_subtree(xml_string: str, root_body_name: str) -> str:
    """
    Extract ONLY robot kinematics subtree from MJCF:
    - Keeps bodies, joints, inertials
    - Removes ALL geoms, cameras, sites
    - Removes non-robot MJCF sections
    """

    tree = ET.ElementTree(ET.fromstring(xml_string))
    root = tree.getroot()

    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("No <worldbody> found in MJCF")

    # ---------------------------
    # 1. find root body recursively
    # ---------------------------
    def find_body(node, name):
        if node.attrib.get("name") == name:
            return node
        for b in node.findall("body"):
            res = find_body(b, name)
            if res is not None:
                return res
        return None

    target = None
    for body in worldbody.findall("body"):
        target = find_body(body, root_body_name)
        if target is not None:
            break

    if target is None:
        raise ValueError(f"Body '{root_body_name}' not found")

    robot = deepcopy(target)

    # ---------------------------
    # 2. recursive cleanup (IMPORTANT PART)
    # ---------------------------
    REMOVE_TAGS = {"geom", "camera", "site", "light"}

    def clean(node):
        # remove unwanted child elements
        for tag in REMOVE_TAGS:
            for child in list(node.findall(tag)):
                node.remove(child)

        # recurse into bodies
        for b in node.findall("body"):
            clean(b)

    clean(robot)

    # ---------------------------
    # 3. replace worldbody
    # ---------------------------
    worldbody.clear()
    worldbody.append(robot)

    # ---------------------------
    # 4. strip global MJCF sections
    # ---------------------------
    REMOVE_TOPLEVEL = [
        "asset",
        "visual",
        "sensor",
        "contact",
        "equality",
        "compiler",
        "option",
        "size",
        "actuator"
    ]

    for tag in REMOVE_TOPLEVEL:
        elem = root.find(tag)
        if elem is not None:
            root.remove(elem)

    return ET.tostring(root, encoding="unicode")

def extract_torque_control_limits(xml_path: str):
    """
    Extract torque control limits (ctrlrange) from MuJoCo XML.

    Args:
        xml_path (str): Path to MuJoCo XML file.

    Returns:
        dict: {actuator_name: (min_torque, max_torque)}
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    limits = {}

    for actuator in root.findall(".//actuator/general"):
        name = actuator.attrib.get("name")

        if "ctrlrange" in actuator.attrib:
            ctrl_min, ctrl_max = map(float, actuator.attrib["ctrlrange"].split())
            limits[name] = (ctrl_min, ctrl_max)

    return limits

class RoboDynamicsModel:
    def __init__(self, urdf_fp, armature, ee_link="gripper0_eef", sim=None):
        # self.parsed_urdf_model = URDF.from_xml_file(urdf_fp) # parsed urdf model for convenience
        
        # self.model, _, _ = pinocchio.buildModelsFromUrdf(
        #     filename=urdf_fp,
        # )
        assert sim is not None, "Must provide sim for RoboDynamicsModel to extract robot subtree and pass to Pinocchio" # TODO: clean up so we only use sim dependency; we keep it like this for now for backwards compatibility with previous projects
        xml_string = sim.model.get_xml() if sim is not None else None
        xml_string = extract_robot_subtree(xml_string, "robot0_base")
    
            
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".xml",
            delete=True   # auto cleanup when context exits
        ) as tmp:

            # write MuJoCo model XML into temp file
            tmp.write(xml_string)
            tmp.flush()   # important: force write to disk

            # pass temp file path to Pinocchio
            self.model, _, _ = pinocchio.buildModelsFromMJCF(
                filename=tmp.name,
            )
            self.torque_limits_ = extract_torque_control_limits(tmp.name)
            
        self.data = self.model.createData()
        self.ee_link_frame_id = self.model.getFrameId(ee_link)
        # assert np.issubdtype(armature.dtype, np.floating) and np.all(armature >= 0) and armature.size == self.model.nq
        
        # self.model.armature[:] = armature[:] 

        self.base_pos = None
        self.base_ori = None
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

        self.compute_eef_pose(q)
        self.compute_eef_jacobian(q)
        self.compute_mass_matrix(q)
        self.compute_coriolis_matrix(q, qd)
        self.compute_gravity_torque(q)
        self.compute_operational_space_matrices()
        self.compute_inertia_weight_jac_psuedo_inv()
        self.compute_nullspace_matrix()

    def compute_eef_pose(self, q):
        eef_se3 = self.data.oMf[self.ee_link_frame_id]
        self.ee_pos[:] = eef_se3.translation[:]
        self.ee_ori[:] = eef_se3.rotation[:]

        if self.base_pos is not None:
            np.add(self.ee_pos, self.base_pos, out=self.ee_pos)
        if self.base_ori is not None:
            np.dot(self.base_ori, self.ee_ori, out=self.ee_ori)

    def compute_eef_jacobian(self, q):
        self.J_full[:] = pinocchio.computeFrameJacobian(
            self.model, self.data, q, self.ee_link_frame_id, 
            pinocchio.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )[:]

        self.J_pos[:] = self.J_full[:3]
        self.J_ori[:] = self.J_full[3:]

    def compute_mass_matrix(self, q):
        pinocchio.crba(self.model, self.data, q)
        self.mass_matrix[:] = self.data.M[:]
        self.mass_matrix_inv[:] = inverse_cholesky(self.mass_matrix)[:]

    def compute_coriolis_matrix(self, q, qd):
        pinocchio.computeCoriolisMatrix(self.model, self.data, q, qd)
        self.coriolis_matrix[:] = self.data.C[:]

    def compute_gravity_torque(self, q):
        pinocchio.computeGeneralizedGravity(self.model, self.data, q)
        self.torque_gravity[:] = self.data.g[:]

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
                self.torque_limits_[actuator_name][0]
                for actuator_name in sorted(self.torque_limits_.keys())
            ]
        )

        upper_limit =  np.array(
            [
                self.torque_limits_[actuator_name][1]
                for actuator_name in sorted(self.torque_limits_.keys())
            ]
        )

        return lower_limit, upper_limit
    
    @property 
    def n_dof(self):
        return self.model.nq
    
    def update_base_pose(self, base_pos, base_ori):
        self.base_pos = base_pos
        if not np.isclose(base_ori, IDENT_QUAT).all():
            self.base_ori = T.quat2mat(base_ori)