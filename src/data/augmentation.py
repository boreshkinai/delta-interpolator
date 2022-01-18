import torch
from typing import Optional, List

from src.data.types import DataTypes
from src.geometry.motion_utils import find_Yrotation_to_align_with_Xplus
from src.geometry.quaternions import remove_quat_discontinuities
from src.geometry.rotations import get_random_rotation_around_axis, get_random_rotation_matrices_around_random_axis
from src.geometry.skeleton import Skeleton

from pytorch3d.transforms import quaternion_multiply, quaternion_apply


def feature(func):
    func.is_feature = True
    return func


def get_view_shape(dt, tensor):
    if dt not in DataTypes:
        raise Exception("datatype " + dt + " not supported")

    denom = DataTypes[dt]
    assert denom > 0, "invalid number of elements in datatype"
    return (-1, int(tensor.shape[1] / denom), denom)


class BaseAugmentation:
    def __init__(self, features=None) -> None:
        self.features = features

    def _get_callback(self, dt):
        if dt == "Scalar":
            return self.scalar
        elif dt == "Vector2":
            return self.vector2
        elif dt == "Vector3":
            return self.vector3
        elif dt == "Quaternion":
            return self.quaternion
        else:
            assert False, "type not yet supported"

    def init(self, dataset, features):
        self.dataset = dataset
        if self.features is None:
            self.features = features
        assert all(x in features for x in self.features), "one or more feature specified does not exist in the dataset"

        self.callstack = []
        for f in self.features:
            datatype, tensor, slice_indices = self.dataset.get_feature(f)
            self.callstack.append((get_view_shape(datatype, tensor), slice_indices, self._get_callback(datatype), f))

    def transform(self, flattened_batch):
        self.begin_batch(flattened_batch)
        for q in self.callstack:
            view = q[0]
            narrow = q[1]
            callback = q[2]
            feature_name = q[3]
            slice = flattened_batch.narrow(1, narrow[0], narrow[1] - narrow[0])
            reshaped = slice.view(view[0], view[1], view[2])
            if feature_name is None:
                modified = callback(reshaped)
            else:
                modified = callback(reshaped, feature_name=feature_name)
            #assert modified.shape == reshaped.shape, "augmentation should not change the shape of the returned tensors"
            reshaped.copy_(modified)
        return flattened_batch

    def begin_batch(self, batch):
        pass

    def compute(self, batch):
        pass

    def scalar(self, scalar_tensor: torch.Tensor, feature_name: Optional[str] = None) -> torch.Tensor:
        return scalar_tensor

    def vector2(self, vector2_tensor: torch.Tensor, feature_name: Optional[str] = None) -> torch.Tensor:
        return vector2_tensor

    def vector3(self, vector3_tensor: torch.Tensor, feature_name: Optional[str] = None) -> torch.Tensor:
        return vector3_tensor

    def quaternion(self, quaterion_tensor: torch.Tensor, feature_name: Optional[str] = None) -> torch.Tensor:
        return quaterion_tensor


class FeatureAugmentation(BaseAugmentation):
    def __init__(self):
        super().__init__(None)

    def init(self, dataset, features):
        self.dataset = dataset
        if self.features is None:
            self.features = features
        assert all(x in features for x in self.features), "one or more feature specified does not exist in the dataset"

        # methods matching a feature name
        object_methods = [getattr(self, method_name) for method_name in dir(self) if callable(getattr(self, method_name)) and method_name in features]
        object_methods = [o for o in object_methods if o.is_feature is True]
        if len(object_methods)==0:
            raise Exception("this FeatureAugmentation does not declare a method which matches a feature name in the dataset")

        # check that all @feature attributes match a known feature
        all_methods_with_feature_attribute = [method_name for method_name in dir(self) if callable(getattr(self, method_name)) and hasattr(getattr(self, method_name), "is_feature") ]
        assert all(x in features for x in all_methods_with_feature_attribute), "one or more method is decorated with the @feature attribute but the feature is not found in the dataset"

        self.callstack = []
        for f in object_methods:
            datatype, tensor, slice_indices = self.dataset.get_feature(f.__name__)
            self.callstack.append((get_view_shape(datatype, tensor), slice_indices, f, None))


class BatchRemoveQuatDiscontinuities(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, batch):
        quaternions = batch['joint_rotations'].clone()
        quaternions = remove_quat_discontinuities(quaternions)
        batch['joint_rotations'] = quaternions
        return batch


class BatchYRotateOnFrame(torch.nn.Module):
    def __init__(self, skeleton, rotation_frame=0):
        super().__init__()
        self.skeleton = skeleton
        self.rotation_frame = rotation_frame

    def forward(self, batch):
        positions = batch['joint_positions'].clone()
        rotations = batch['joint_rotations'].clone()

        rotate = Rotate(skeleton=self.skeleton)

        # Get root Y rotation on particular frame
        root_rotation = rotations[..., self.rotation_frame, rotate.root_idx[0], :].clone()

        batch_size, n_frames, n_joints = positions.shape[:3]

        # Find the rotation to use
        y_rotation_quat = find_Yrotation_to_align_with_Xplus(root_rotation)

        y_rotation_quat = y_rotation_quat.unsqueeze(1)  # Restore frames dim
        y_rotation_quat_root = y_rotation_quat.expand(-1, positions.shape[1], -1).unsqueeze(2)
        y_rotation_quat = y_rotation_quat_root.expand(-1, -1, positions.shape[2], -1).contiguous()  # Restore bone dim

        # Apply rotation
        positions, _ = rotate.forward(joint_positions=positions,
                                      joint_rotations=None,
                                      rotation_matrix=None,
                                      rotation_quat=y_rotation_quat)

        _, rotations = rotate.forward(joint_positions=None,
                                      joint_rotations=rotations.view(batch_size * n_frames, n_joints, 4),
                                      rotation_matrix=None,
                                      rotation_quat=y_rotation_quat[..., rotate.root_idx, :].view(batch_size * n_frames,
                                                                                                  4))

        positions = positions.view(batch_size, n_frames, n_joints, 3)
        rotations = rotations.view(batch_size, n_frames, n_joints, 4)

        batch['joint_positions'] = positions
        batch['joint_rotations'] = rotations

        return batch


class BatchCenterXZ(torch.nn.Module):
    def __init__(self, local_positions=False):
        super().__init__()
        self.local = local_positions

    def forward(self, batch):
        positions = batch['joint_positions'].clone()
        xzs = torch.mean(positions[..., 0:1, ::2], dim=1, keepdim=True)
        if self.local:
            positions[..., 0:1, 0] = positions[..., 0:1, 0] - xzs[..., 0]
            positions[..., 0:1, 2] = positions[..., 0:1, 2] - xzs[..., 1]
        else:
            positions[..., :, 0] = positions[..., :, 0] - xzs[..., 0]
            positions[..., :, 2] = positions[..., :, 2] - xzs[..., 1]
        batch['joint_positions'] = positions
        return batch


class BatchRotate(torch.nn.Module):
    def __init__(self, skeleton, axis=None):
        super().__init__()
        if axis is None:
            axis = [0.0, 1.0, 0.0]
        self.rotating = Rotate(skeleton=skeleton, axis=axis)
        self.nb_joints = skeleton.nb_joints

    def forward(self, batch):
        batch_length = batch['joint_positions'].shape[1]
        batch_size = batch['joint_positions'].shape[0]

        # Single rotation per sequence
        rot_mat, rot_quat = self.rotating.generate_random_rotations(batch_size)
        rot_mat = rot_mat.unsqueeze(1).repeat(1, batch_length, 1, 1).view(-1, 3, 3)
        rot_quat = rot_quat.unsqueeze(1).repeat(1, batch_length, 1).view(-1, 4)
        original_positions = batch['joint_positions'].reshape(-1, self.nb_joints, 3)
        original_rotations = batch['joint_rotations'].reshape(-1, self.nb_joints, 4)
        rotated_positions, rotated_rotations = self.rotating.forward(original_positions, original_rotations,
                                                                     rot_mat, rot_quat)
        batch['joint_positions'] = rotated_positions.view(batch_size, -1, self.nb_joints, 3)
        batch['joint_rotations'] = rotated_rotations.view(batch_size, -1, self.nb_joints, 4)
        return batch


class BatchMirror(torch.nn.Module):
    def __init__(self, skeleton, mirror_prob=0.5):
        super().__init__()
        self.mirroring = Mirror(skeleton=skeleton)
        self.mirror_probability = mirror_prob
        self.nb_joints = skeleton.nb_joints

    def forward(self, batch):
        if torch.rand(1)[0] < self.mirror_probability:
            batch_size = batch['joint_positions'].shape[0]
            original_positions = batch['joint_positions'].reshape(-1, self.nb_joints, 3)
            original_rotations = batch['joint_rotations'].reshape(-1, self.nb_joints, 4)
            mirrored_positions, mirrored_rotations = self.mirroring.forward(original_positions, original_rotations)
            batch['joint_positions'] = mirrored_positions.view(batch_size, -1, self.nb_joints, 3)
            batch['joint_rotations'] = mirrored_rotations.view(batch_size, -1, self.nb_joints, 4)
        return batch


class Rotate(torch.nn.Module):
    def __init__(self, skeleton: Skeleton, axis: Optional[List[float]] = None):
        super().__init__()

        # only root joints are rotated as all others are in local space
        self.register_buffer('root_idx', torch.LongTensor([0]))

        if axis is None:
            self.rotation_axis = None
        else:
            self.register_buffer('rotation_axis', torch.FloatTensor(axis).view(1, 3))

    def generate_random_rotations(self, count: int):
        if self.rotation_axis is None:
            axis = torch.randn((count, 3), device=self.root_idx.device)
            random_rot, random_quat = get_random_rotation_around_axis(axis, return_quaternion=True)
        else:
            random_rot, random_quat = get_random_rotation_around_axis(self.rotation_axis.repeat(count, 1), return_quaternion=True)
        return random_rot, random_quat

    def forward(self, joint_positions: torch.Tensor = None, joint_rotations: torch.Tensor = None, rotation_matrix = None, rotation_quat = None):
        if joint_positions is None:
            new_positions = None
        else:
            if rotation_matrix is not None:
                new_positions = torch.matmul(rotation_matrix.type_as(joint_positions), joint_positions.transpose(2, 1)).transpose(2, 1)
            elif rotation_quat is not None:
                new_positions = quaternion_apply(rotation_quat.type_as(joint_positions), joint_positions)
            else:
                raise ValueError("Missing a rotation element!")

            new_positions = new_positions.contiguous()

        if joint_rotations is None:
            new_rotations = None
        else:
            quats = rotation_quat.type_as(joint_rotations)
            if quats.shape[0] == 1:
                quats = quats.repeat(joint_rotations.shape[0], 1)
            else:
                quats = quats.repeat_interleave(len(self.root_idx), dim=0)
            new_rotations = joint_rotations.clone()
            new_rotations[:, self.root_idx, :] = quaternion_multiply(quats, joint_rotations[:, self.root_idx, :].view(-1, 4)).view(-1, self.root_idx.shape[0], 4)
            new_rotations = new_rotations.view_as(joint_rotations).contiguous()

        return new_positions, new_rotations


class Mirror(torch.nn.Module):
    def __init__(self, skeleton: Skeleton):
        super().__init__()
        self.register_buffer('reflection_matrix', torch.eye(3))
        self.reflection_matrix[0, 0] = -1

        # quaternions must be in w, x, y, z order
        self.register_buffer('quat_indices', torch.tensor([2, 3]))
        self.register_buffer('swap_index_list', torch.LongTensor(skeleton.bone_pair_indices))

    def forward(self, joint_positions: torch.Tensor = None, joint_rotations: torch.Tensor = None):
        if joint_positions is None:
            new_positions = None
        else:
            new_positions = self._swap_tensor(joint_positions)
            new_positions = torch.matmul(self.reflection_matrix, new_positions.permute(1, 2, 0)).permute(2, 0, 1)
            new_positions = new_positions.contiguous()

        if joint_rotations is None:
            new_rotations = None
        else:
            new_rotations = self._swap_tensor(joint_rotations)
            new_rotations[:, :, self.quat_indices] *= -1
            new_rotations = new_rotations.contiguous()

        return new_positions, new_rotations

    def _swap_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor[:, self.swap_index_list, :].view_as(tensor)


class RandomTranslation(BaseAugmentation):
    axis = torch.FloatTensor([1.0, 1.0, 1.0])
    range = [-1.0, 1.0]
    random_vec = None

    def __init__(self, axis: Optional[list] = None, range: Optional[list] = None, features: Optional[list] = None):
        super().__init__(features)
        if axis is not None:
            self.axis = torch.FloatTensor(axis)
        if range is not None:
            self.range = torch.FloatTensor(range)

    def begin_batch(self, batch):
        self.random_vec = torch.FloatTensor(batch.shape[0], 3).uniform_(self.range[0], self.range[1]) * self.axis
        self.random_vec = self.random_vec.type_as(batch)

    def vector3(self, vec3, feature_name=None):
        p = vec3.view(vec3.shape[0], -1) + self.random_vec.repeat(1, vec3.shape[1])
        return p.view_as(vec3)


class RandomRotation(BaseAugmentation):
    rotation_axis = None
    random_rot = None
    random_quat = None

    def __init__(self, axis: Optional[list] = None, features: Optional[list] = None):
        super().__init__(features)
        if axis is not None:
            self.rotation_axis = torch.FloatTensor(axis).view(1, 3)

    def begin_batch(self, batch):
        if self.rotation_axis is None:
            self.random_rot, self.random_quat = get_random_rotation_matrices_around_random_axis(batch, return_quaternion=True)
        else:
            self.random_rot, self.random_quat = get_random_rotation_around_axis(
                self.rotation_axis.repeat(batch.shape[0], 1), return_quaternion=True)

        self.random_rot = self.random_rot.type_as(batch)
        self.random_quat = self.random_quat.type_as(batch)  # order is w, x, y, z

    def vector3(self, vector3_tensor: torch.Tensor, feature_name=None) -> torch.Tensor:
        new_vec3 = torch.matmul(self.random_rot, vector3_tensor.transpose(2, 1)).transpose(2, 1)
        return new_vec3

    def quaternion(self, quat4_tensor: torch.Tensor, feature_name=None) -> torch.Tensor:
        # The hip quaternion will always be the first index of the second axis
        # Rotations are all local wrt to parent so we need to only rotate hip bone
        # input / output quat order is x, y, z, w
        quat4_tensor[:, 0] = quaternion_multiply(self.random_quat, quat4_tensor[:, 0, [3, 0, 1, 2]])[:, [1, 2, 3, 0]]
        return quat4_tensor


class MirrorSkeleton(BaseAugmentation):
    axis = [1.0, 0.0, 0.0]
    skeleton = None
    reflection_matrix_2 = None
    reflection_matrix_3 = None
    quat_indices = None
    mirror = False

    def __init__(self, skeleton: Skeleton, axis: Optional[list] = None, features: Optional[list] = None):
        super().__init__(features)
        self.axis = [float(i) for i in axis] if axis is not None else self.axis
        self._build_reflection_matrices(self.axis)
        self.swap_index_list = torch.LongTensor(skeleton.bone_pair_indices)

    def begin_batch(self, batch):
        self.swap_index_list = self.swap_index_list.to(batch.device)
        if self.reflection_matrix_2 is not None:
            self.reflection_matrix_2 = self.reflection_matrix_2.type_as(batch)
        if self.reflection_matrix_3 is not None:
            self.reflection_matrix_3 = self.reflection_matrix_3.type_as(batch)

        self.mirror = False
        if torch.rand(1)[0] > 0.5:
            self.mirror = True

    def _build_reflection_matrices(self, axis: list):
        assert len(axis) >= 2 and len(axis) <= 3, "Please ensure the specified mirror axis is either 2 dimensional or 3 dimensional"
        axis2D = axis[:2] if len(axis) == 3 else axis
        axis3D = axis

        self.reflection_matrix_2 = None
        self.reflection_matrix_3 = None

        if len(axis2D) == 2:
            self.reflection_matrix_2 = torch.eye(2)
            if axis2D == [0.0, 1.0]:
                # Reflect about Y
                self.reflection_matrix_2[0, 0] *= -1
            elif axis2D == [1.0, 0.0]:
                # Reflect about X
                self.reflection_matrix_2[1, 1] *= -1
            else:
                if len(axis3D) != 3:
                    # If only 2D axis passed we know it should be correct
                    raise ValueError("Invalid axis for 2D mirroring, please use one of: \n {}".format(
                        torch.eye(2).cpu().detach().numpy()))

        if len(axis3D) == 3:
            # Reflection for vector3
            self.reflection_matrix_3 = torch.eye(3)
            if axis3D == [1.0, 0.0, 0.0]:
                # YZ plane
                self.reflection_matrix_3[0, 0] *= -1
                self.quat_indices = torch.tensor([1, 2])
            else:
                raise ValueError("Invalid axis for 3D mirroring, please use one of: \n {}".format(
                    torch.eye(3)[-2].cpu().detach().numpy()))

    def _flip_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(tensor).index_copy_(1, self.swap_index_list, tensor)

    def vector2(self, vector2_tensor: torch.Tensor, feature_name=None) -> torch.Tensor:
        if self.reflection_matrix_2 is None:
            return vector2_tensor

        if not self.mirror:
            return vector2_tensor
        flipped = self._flip_tensor(vector2_tensor)
        return torch.matmul(self.reflection_matrix_2, flipped.transpose(2, 1)).transpose(1, 2)

    def vector3(self, vector3_tensor: torch.Tensor, feature_name=None) -> torch.Tensor:
        if self.reflection_matrix_3 is None:
            return vector3_tensor

        if not self.mirror:
            return vector3_tensor
        flipped = self._flip_tensor(vector3_tensor)
        return torch.matmul(self.reflection_matrix_3, flipped.permute(1, 2, 0)).permute(2, 0, 1)

    def quaternion(self, quat4_tensor: torch.Tensor, feature_name=None) -> torch.Tensor:
        if self.quat_indices is None:
            return quat4_tensor

        if not self.mirror:
            return quat4_tensor
        flipped = self._flip_tensor(quat4_tensor)
        flipped[:, :, self.quat_indices] *= -1
        return flipped