import torch

from src.modules.layers import (
    Embedding,
    LayerNorm,
    FCBlock, MultiHeadAttention
)
from typing import Tuple, Dict
from src.modules.interpolator import Interpolator


class ResidualBlock(torch.nn.Module):
    """Fully connected residual block"""

    def __init__(self, num_layers: int, layer_width: int, dropout: float, size_in: int):
        super(ResidualBlock, self).__init__()
        self.num_layers = num_layers
        self.layer_width = layer_width

        self.fc_layers = [torch.nn.Linear(size_in, layer_width)]
        self.relu_layers = [torch.nn.LeakyReLU(inplace=True)]
        if dropout > 0.0:
            self.fc_layers.append(torch.nn.Dropout(p=dropout))
            self.relu_layers.append(torch.nn.Identity())
        self.fc_layers += [torch.nn.Linear(layer_width, layer_width) for _ in range(num_layers - 1)]
        self.relu_layers += [torch.nn.LeakyReLU(inplace=True) for _ in range(num_layers - 1)]

        self.fc_layers = torch.nn.ModuleList(self.fc_layers)
        self.relu_layers = torch.nn.ModuleList(self.relu_layers)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = x
        for layer, relu in zip(self.fc_layers, self.relu_layers):
            h = relu(layer(h))
        return h + x
    

class Transformer(torch.nn.Module):
    """
    Fully-connected residual architechture with many categorical inputs wrapped in embeddings
    """

    def __init__(self,
                 num_blocks_enc: int, num_layers_enc: int, layer_width_enc,
                 num_blocks_dec: int, num_layers_dec: int, layer_width_dec: int,
                 dropout: float,
                 size_in: int, size_out: int, size_out_stage1: int, nb_joints: int,
                 embedding_dim: int, embedding_size: int, embedding_num: int, 
                 layer_norm: bool = True, eps: float = 1e-6,
                 num_heads: int = 8, delta_mode: str = 'interpolator', 
                 input_delta_mode: str = 'last_pose'):
        super().__init__()

        self.layer_width_enc = layer_width_enc
        self.nb_joints = nb_joints
        self.size_in = size_in
        self.size_out = size_out
        self.size_out_stage1 = size_out_stage1
        self.num_heads = num_heads
        self.num_blocks_dec = num_blocks_dec
            
        if delta_mode in ['interpolator', 'last_pose', 'none']:
            self.delta_mode = delta_mode
        else:
            raise ValueError(f"Delta mode {delta_mode} is not implemented")
            
        if input_delta_mode in ['last_pose', 'none']:
            self.input_delta_mode = input_delta_mode
        else:
            raise ValueError(f"Inpuit Delta mode {input_delta_mode} is not implemented")
            
        self.input_projection_src = torch.nn.Linear(size_in*nb_joints + embedding_dim * embedding_num, layer_width_enc)
        self.input_projection_tgt = torch.nn.Linear(size_in*nb_joints + embedding_dim * embedding_num, layer_width_enc)
        
        self.interpolator = Interpolator(nb_joints, space='ortho6d')

        self.embeddings = [Embedding(num_embeddings=embedding_size, embedding_dim=embedding_dim) for _ in
                           range(embedding_num)]
        
        self.encoder_blocks = [ResidualBlock(num_layers=num_layers_enc, layer_width=layer_width_enc, 
                                             dropout=dropout, size_in=layer_width_enc) for _ in range(num_blocks_enc)]
        
        self.mha = [MultiHeadAttention(in_features=layer_width_enc, head_num=self.num_heads,
                                       activation=None) for _ in range(num_blocks_enc)]
        self.layer_norm = [LayerNorm(num_features=layer_width_enc) for _ in range(num_blocks_enc)]
        
        self.stage1_blocks = FCBlock(num_layers=num_layers_dec,
                                     layer_width=layer_width_dec,
                                     dropout=dropout,
                                     size_in=layer_width_dec,
                                     size_out=size_out_stage1*nb_joints)
        self.stage2_blocks = FCBlock(num_layers=num_layers_dec,
                                     layer_width=layer_width_dec,
                                     dropout=dropout,
                                     size_in=layer_width_enc,
                                     size_out=size_out*nb_joints)

        self.model = torch.nn.ModuleList(self.encoder_blocks + self.embeddings + \
                                         self.mha + self.layer_norm + \
                                         [self.input_projection_src] + [self.input_projection_tgt])
        

    def decode(self, pose_embedding: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        inputs: 
        pose_embedding torch.Tensor : BxTxC
        """
        batch, n_frames = pose_embedding.shape[:2]
        
        _, stage1_forecast = self.stage1_blocks(pose_embedding)
        _, stage2_forecast = self.stage2_blocks(pose_embedding)

        return stage1_forecast.view(batch, n_frames, self.nb_joints, self.size_out_stage1), \
               stage2_forecast.view(batch, n_frames, self.nb_joints, self.size_out)


    def forward(self, input_data: Dict[str, torch.Tensor], *args) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x the continuous input : BxTxNxC
        """        
        joint_positions_interpolator, joint_rotations_interpolator = self.interpolator(input_data)
        
        joint_positions_ref = input_data["joint_positions_global"][:, -2, 0, :].unsqueeze(1).unsqueeze(2)
        joint_rotations_ref = input_data["joint_rotations_ortho6d"][:, -2, 0, :].unsqueeze(1).unsqueeze(2)
        
        if self.input_delta_mode == 'none':
            x_src = torch.cat([input_data["joint_rotations_ortho6d"], 
                               input_data["joint_positions_global"]], 
                              dim=-1)
            x_src = x_src.view(*x_src.shape[:2], -1)
            x_tgt = torch.cat([joint_rotations_interpolator, 
                               joint_positions_interpolator], 
                              dim=-1)
            x_tgt = x_tgt.view(*x_tgt.shape[:2], -1)
        elif self.input_delta_mode == 'last_pose':
            # The input is derived by concatenating both rotations and positions and then flattenning over the joints dimension
            x_src = torch.cat([input_data["joint_rotations_ortho6d"] - joint_rotations_ref, 
                               input_data["joint_positions_global"] - joint_positions_ref], 
                              dim=-1)
            x_src = x_src.view(*x_src.shape[:2], -1)
            x_tgt = 0.0 * torch.cat([joint_rotations_interpolator - joint_rotations_ref, 
                                     joint_positions_interpolator - joint_positions_ref], 
                                    dim=-1)
            x_tgt = x_tgt.view(*x_tgt.shape[:2], -1)
        
        # This is the time vector
        src_time = input_data["input_frame_indices"].to(dtype=torch.int64)
        tgt_time = input_data["target_frame_indices"].to(dtype=torch.int64)
        
        
        if self.delta_mode == 'interpolator':
            joint_rotations_interpolator = torch.cat([joint_rotations_interpolator[:, :1].repeat([1, src_time.shape[0]-1, 1, 1]),
                                                      joint_rotations_interpolator,
                                                      joint_rotations_interpolator[:, -1:]], dim=1)
            joint_positions_interpolator = torch.cat([joint_positions_interpolator[:, :1].repeat([1, src_time.shape[0]-1, 1, 1]),
                                                      joint_positions_interpolator,
                                                      joint_positions_interpolator[:, -1:]], dim=1)
        elif self.delta_mode == 'last_pose':
            joint_rotations_interpolator = input_data["joint_rotations_ortho6d"][:, -2, :, :].unsqueeze(1)
            joint_rotations_interpolator = joint_rotations_interpolator.repeat([1, src_time.shape[0] + tgt_time.shape[0], 1, 1])
            
            joint_positions_interpolator = input_data["joint_positions_global"][:, -2, :, :].unsqueeze(1)
            joint_positions_interpolator = joint_positions_interpolator.repeat([1, src_time.shape[0] + tgt_time.shape[0], 1, 1])
        elif self.delta_mode == 'none':
            joint_rotations_interpolator = 0.0
            joint_positions_interpolator = 0.0
            
            
        src_time_batch = src_time.repeat([x_src.shape[0], 1])
        length = tgt_time.shape[0] * torch.ones_like(src_time_batch)
        ee_src = [x_src]
        for i, v in enumerate([src_time_batch, length]):
            ee_src.append(self.embeddings[i](v))
        x_src = torch.cat(ee_src, dim=-1)
        
        ee_tgt = [x_tgt]
        tgt_time_batch = tgt_time.repeat([x_src.shape[0], 1])
        length = tgt_time.shape[0] * torch.ones_like(tgt_time_batch)
        for i, v in enumerate([tgt_time_batch, length]):
            ee_tgt.append(self.embeddings[i](v))
        x_tgt = torch.cat(ee_tgt, dim=-1)
        
        x_src = self.input_projection_src(x_src)
        x_tgt = self.input_projection_tgt(x_tgt)
        for i, block in enumerate(self.encoder_blocks):
            shortcut_src = x_src
            x_src = self.mha[i](q=x_src, k=x_src, v=x_src)
            x_src = self.layer_norm[i](x_src + shortcut_src)
            x_src = torch.relu(x_src)
            x_src = block(x_src)
            
            shortcut_tgt = x_tgt
            x_tgt = self.mha[i](q=x_tgt, k=x_src, v=x_src)
            x_tgt = self.layer_norm[i](x_tgt + shortcut_tgt)
            x_tgt = torch.relu(x_tgt)
            x_tgt = block(x_tgt)
            
        pose_embedding = torch.zeros((x_src.shape[0], src_time.shape[0] + tgt_time.shape[0], x_src.shape[2])).to(x_src)
        pose_embedding[:, src_time, ...] = x_src
        pose_embedding[:, tgt_time, ...] = x_tgt
        
        stage1, stage2 = self.decode(pose_embedding)
        stage1 = stage1 + joint_positions_interpolator
        stage2 = stage2 + joint_rotations_interpolator
        
        return stage1, stage2
