import torch
from src.geometry.vector import normalize_vector


def remove_quat_discontinuities(rotations):
    rotations = rotations.clone()
    rots_inv = -rotations
    for i in range(1, rotations.shape[1]):
        replace_mask = torch.sum(rotations[:, i-1:i, ...] * rotations[:, i:i+1, ...], 
                                 dim=-1, keepdim=True) < \
                       torch.sum(rotations[:, i-1:i, ...] * rots_inv[:, i:i+1, ...], 
                                 dim=-1, keepdim=True)
        replace_mask = replace_mask.squeeze(1).type_as(rotations)
        rotations[:, i, ...] = replace_mask * rots_inv[:, i, ...] + (1.0 - replace_mask) * rotations[:, i, ...]
    return rotations


# returns quaternion so that v_from rotated by this quaternion equals v_to
# v_... are vectors of size (..., 3)
# returns quaternion in w, x, y, z order, of size (..., 4)
# note: such a rotation is not unique, there is an infinite number of solutions
# this implementation returns the shortest arc
def from_to_quaternion(v_from, v_to, eps: float = 1e-8):
    v_from_unit = normalize_vector(v_from, eps=eps)
    v_to_unit = normalize_vector(v_to, eps=eps)

    w = (v_from_unit * v_to_unit).sum(dim=1) + 1
    xyz = torch.cross(v_from_unit, v_to_unit, dim=1)
    q = torch.cat([w.unsqueeze(1), xyz], dim=1)
    return normalize_vector(q, eps=eps)


def slerp(q0, q1, t):
    """
    Spherical Linear Interpolation of quaternions
    https://www.euclideanspace.com/maths/algebra/realNormedAlgebra/quaternions/slerp/index.htm
    :param q0: Start quats (w, x, y, z) : shape = (B, J, 4)
    :param q1: End quats (w, x, y, z) : shape = (B, J, 4)
    :param t:  Step (in [0, 1]) : shape = (B, T, J, 1)
    :return: Interpolated quat (w, x, y, z) : shape = (B, J, 4)
    """
    q0 = q0.unsqueeze(1)
    q1 = q1.unsqueeze(1)
    
    # Dot product
    cos_half_theta = torch.sum(q0 * q1, dim=-1, keepdim=True)

    # Make sure we take the shortest path :
    q1_antipodal = -q1
    q1 = torch.where(cos_half_theta > 0, q1, q1_antipodal)
    
    half_theta = torch.acos(cos_half_theta)
    # torch.sin must be safer here
    sin_half_theta = torch.sqrt(1.0 - cos_half_theta * cos_half_theta)
    ratio_a = torch.sin((1 - t) * half_theta) / sin_half_theta
    ratio_b = torch.sin(t * half_theta) / sin_half_theta
    
    qt = ratio_a * q0 + ratio_b * q1    
    # If the angle was constant, prevent nans by picking the original quat:
    qt = torch.where(torch.abs(cos_half_theta) >= 1.0, q0, qt)
    return qt
