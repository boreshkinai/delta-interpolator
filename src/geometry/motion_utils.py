import numpy as np
import torch
from pytorch3d.transforms import quaternion_apply

from src.geometry.quaternions import from_to_quaternion
from src.geometry.vector import normalize_vector


def lerp(p0, p1, t):
    pt = (1.0 - t) * p0 + t * p1
    return pt


def find_Yrotation_to_align_with_Xplus(q):
    """

    :param q: Quats tensor for current rotations (B, 4)
    :return y_rotation: Quats tensor of rotations to apply to q to align with X+
    """
    mask = torch.tensor(np.array([[1.0, 0.0, 1.0]]), dtype=torch.float).expand(q.shape[0], -1)
    forward = mask * quaternion_apply(q, torch.tensor(np.array([[-1.0, 0.0, 0.0]]), dtype=torch.float).expand(q.shape[0], -1))
    forward = normalize_vector(forward)
    y_rotation = normalize_vector(from_to_quaternion(forward, torch.tensor(np.array([[1, 0, 0]]), dtype=torch.float).expand(q.shape[0], -1)))
    return y_rotation