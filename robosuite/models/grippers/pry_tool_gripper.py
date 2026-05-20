"""
Null Gripper (if we don't want to attach gripper to robot eef).
"""
from robosuite.models.grippers.gripper_model import GripperModel
from robosuite.utils.mjcf_utils import xml_path_completion


class PryToolGripper(GripperModel):
    """
    Dummy Gripper class to represent no gripper

    Args:
        idn (int or str): Number or some other unique identification string for this gripper instance
    """

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("grippers/prying_tool_gripper.xml"), idn=idn)

    def format_action(self, action):
        return action

    @property
    def init_qpos(self):
        return None

    @property
    def _important_geoms(self):
        """
        Geoms corresponding to important components of the gripper (by default, left_finger, right_finger,
        left_fingerpad, right_fingerpad).
        Note that these are the raw string names directly pulled from a gripper's corresponding XML file,
        NOT the adjusted name with an auto-generated naming prefix

        Note that this should be a dict of lists.

        Returns:
            dict of list: Raw XML important geoms, where each set of geoms are grouped as a list and are
            organized by keyword string entries into a dict
        """
        return {
            "pry_head_collision": ["pry_head_collision"],
        }