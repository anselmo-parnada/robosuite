import numpy as np
import mujoco

from robosuite.environments.base import MujocoEnv


def is_geom_in_contact(env : MujocoEnv, geom_id : int):
    """
    Check if a geom is in contact with any other geom in the environment

    Args:
        env (MujocoEnv): Current robosuite environment
        geom_id (int): Geom ID to check

    Returns:
        bool: True if geom is in contact with any other geom, False otherwise
    """
    mj_contact_struct = env.sim.data.contact
    return geom_id in mj_contact_struct.geom


def get_contact_forces_on_geom(
        env : MujocoEnv, geom_id : int
    ):
    """
    Get the contact forces acting on a geom

    Args:
        env (MujocoEnv): Current robosuite environment
        geom_id (int): Geom ID to check

    Returns:
        list or None: List of contact forces acting on the geom, or None if the geom is not in contact.
    """
    if not is_geom_in_contact(env, geom_id):
        return None

    mj_contact_struct = env.sim.data.contact
    contact_forces = []
    contact_force = np.zeros(6)

    for i in range(env.sim.data.ncon):
        if geom_id in mj_contact_struct.geom[i]:
            mujoco.mj_contactForce(env.sim.model._model, env.sim.data._data, geom_id, contact_force)
            contact_forces.append(contact_force.copy())

    return contact_forces


def get_largest_contact_force_on_geom(
        env : MujocoEnv, geom_id : int
    ):
    """
    Get the largest contact force acting on a geom

    Args:
        env (MujocoEnv): Current robosuite environment
        geom_id (int): Geom ID to check

    Returns:
        np.ndarray or None: Largest contact force acting on the geom, or None if the geom is not in contact.
    """
    
    contact_forces = get_contact_forces_on_geom(env, geom_id)
    if contact_forces is None:
        return None
    return max(contact_forces, key=lambda x: np.linalg.norm(x))