import torch
from torch import nn
from typing import Tuple, Dict


class ZeroVelocity(nn.Module):
    def __init__(self, size_in: int = 6, size_out: int = 6, nb_joints: int = None):
        super(ZeroVelocity, self).__init__()
        
        self.size_in = size_in
        self.size_out = size_out
        self.nb_joints = nb_joints

    def forward(self, input_data: Dict[str, torch.Tensor]):
        input_timestamps = input_data["input_frame_indices"]
        target_timestamps = input_data["target_frame_indices"]
        
        timestamps_diff = (input_timestamps.reshape(-1, 1) - target_timestamps).to(dtype=torch.float32)
        timestamps_diff = torch.where(timestamps_diff > 0.0, torch.tensor(float("-inf")).to(timestamps_diff), timestamps_diff)
        timestamps_diff = timestamps_diff - timestamps_diff.max(dim=0, keepdim=True)[0]
        
        weights = torch.softmax(1e5 * timestamps_diff, dim=0)

        root_position = torch.matmul(input_data["root_positions"].transpose(-1, -2),
                                     weights.unsqueeze(0)).transpose(-1, -2)
        joint_positions = root_position.unsqueeze(-2).repeat([1, 1, self.nb_joints, 1]).contiguous()
        
        joint_rotations = torch.matmul(input_data["joint_rotations_ortho6d"].transpose(-1, -3), 
                                       weights.unsqueeze(0)).transpose(-1, -3).contiguous()
        
        return joint_positions, joint_rotations
    

class ZeroVelocityWrapper(ZeroVelocity):

    def forward(self, input_data: Dict[str, torch.Tensor], *args) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x the continuous input : BxTxNxC
        """        
        joint_positions_interpolator, joint_rotations_interpolator = super().forward(input_data)
        
        src_time = input_data["input_frame_indices"].to(dtype=torch.int64)       
        
        joint_rotations_interpolator = torch.cat([joint_rotations_interpolator[:, :1].repeat([1, src_time.shape[0]-1, 1, 1]),
                                                  joint_rotations_interpolator,
                                                  joint_rotations_interpolator[:, -1:]], dim=1)
        joint_positions_interpolator = torch.cat([joint_positions_interpolator[:, :1].repeat([1, src_time.shape[0]-1, 1, 1]),
                                                  joint_positions_interpolator,
                                                  joint_positions_interpolator[:, -1:]], dim=1)
        
        return joint_positions_interpolator, joint_rotations_interpolator
