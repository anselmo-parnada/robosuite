"""
Null Gripper (if we don't want to attach gripper to robot eef).
"""
from robosuite.models.grippers.gripper_model import GripperModel
from robosuite.utils.mjcf_utils import xml_path_completion


class Peg3Gripper(GripperModel):
    """
    Dummy Gripper class to represent no gripper

    Args:
        idn (int or str): Number or some other unique identification string for this gripper instance
    """

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("grippers/peg_3_gripper.xml"), idn=idn)

    def format_action(self, action):
        return action

    @property
    def init_qpos(self):
        return None
