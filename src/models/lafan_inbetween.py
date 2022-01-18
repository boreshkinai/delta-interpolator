import errno
import os
import json
import torch
from torch import nn

from pytorch_lightning import LightningModule
import hydra
from hydra.utils import instantiate

from dataclasses import dataclass
from typing import Any, Dict, List, Union

from src.data.frame_sampler import RandomMiddleFramesRemoverOptions
from src.data.sequence_module import SequenceDataModuleOptions

from src.geometry.skeleton import Skeleton
from src.geometry.quaternions import remove_quat_discontinuities

from src.utils.model_factory import ModelFactory
from src.utils.options import BaseOptions
from src.utils.onnx_export import export_named_model_to_onnx

from omegaconf import DictConfig

from pytorch3d.transforms import (
    matrix_to_quaternion,
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
    quaternion_to_matrix
)

from src.evaluation.lafan_benchmark import LafanBenchmarkEvaluator


@dataclass
class LafanInBetweenModelOptions(BaseOptions):
    frame_sampler: Any = RandomMiddleFramesRemoverOptions(min_past_context=10,
                                                          max_past_context=10,
                                                          max_future_context=1,
                                                          min_middle_frames=5,
                                                          max_middle_frames=39)
    dataset: SequenceDataModuleOptions = SequenceDataModuleOptions()
    optimizer: Any = None
    scheduler: Any = None
    backbone: Any = None
    quat_loss_scale: float = 10.0
    pos_loss_l1_scale: float = 10.0
    reconstruction_scale: float = 1.0
    datasets_path: str = "./datasets"
    benchmark: str = "lafan"

        
@ModelFactory.register(LafanInBetweenModelOptions, schema_name="LafanInBetween")
class LafanInBetweenModel(LightningModule):

    def __init__(self, skeleton: Skeleton, opts: LafanInBetweenModelOptions):
        super().__init__()

        assert isinstance(opts,
                          DictConfig), f"opt constructor argument must be of type DictConfig but got {type(opts)} instead."
        assert skeleton is not None, "You must provide a valid skeleton"

        self.save_hyperparameters(opts)
        self.skeleton = skeleton

        self.root_idx = self.get_joint_indices('Hips')
        self.frame_sampler = instantiate(opts.frame_sampler)
        self.net = instantiate(opts.backbone, nb_joints=self.skeleton.nb_joints)
        
        self.evaluator = None
        if opts.benchmark != 'None':
            self.evaluator = LafanBenchmarkEvaluator(datasets_path=hydra.utils.to_absolute_path(opts.datasets_path),
                                                     benchmark=opts.benchmark, 
                                                     device='cpu', 
                                                     verbose=True)

    def forward(self, input_data):
        input_data = self.prepare_inputs(input_data)
        joint_positions, joint_rotations_ortho6d = self.net(input_data=input_data)
        return {
            "joint_positions_global": joint_positions,
            "joint_rotations_ortho6d": joint_rotations_ortho6d,
        }

    def forward_wrapped(self, past_frames, future_frames, target_frames):
        """
        Useful to have to avoid having external object, such as the evaluator recreate this chunk of the shared step
        """
        target_data = self.expand_input(target_frames)

        # Must match what is defined in the task.
        input_data = {
            "past_frame_indices": past_frames['frame_indices'],
            "future_frame_indices": future_frames['frame_indices'],
            "output_frame_indices": target_frames['frame_indices'],
            "past_root_positions": past_frames['root_positions'],
            "future_root_positions": future_frames['root_positions'],
            "past_local_quats": past_frames['joint_rotations'],
            "future_local_quats": future_frames['joint_rotations'],
        }
        
        predicted = self.expand_predictions(self.forward(input_data=input_data))
        return target_data, predicted
 
    def training_step(self, batch, batch_idx):        
        losses = self.shared_step(batch)
        self.log_train_losses(losses)
        return losses["total"]
    
    def evaluate(self):
        metrics = None
        if self.evaluator is not None:
            metrics = self.evaluator.evaluate(self)
        return metrics

    def validation_step(self, batch, batch_idx):        
        if batch_idx == 0:
            metrics = self.evaluate()
            for k, v in metrics.items():
                self.log(f"benchmark/{k}", v, on_step=False, on_epoch=True)
        return None

    def l1_position_loss(self, input_positions, target_positions):
        return nn.functional.l1_loss(input_positions.view(-1, 3), target_positions.view(-1, 3))
    
    def l1_quaternion_loss(self, input_quaternions, target_quaternions):
        return nn.functional.l1_loss(input_quaternions.view(-1, 4), target_quaternions.view(-1, 4))

    def expand_input(self, input_data):
        # Add different data representations to the input
        root_positions = input_data['root_positions']
        joint_rotations = input_data['joint_rotations']
        batch_size, n_frames = root_positions.shape[:2]
        joint_rotations_mtx = quaternion_to_matrix(joint_rotations)
        joint_rotations_ortho6d = matrix_to_rotation_6d(joint_rotations_mtx)

        # Global-space representation
        joint_positions_global, joint_rotations_mtx_global = self.skeleton.forward(
            joint_rotations_mtx.view(-1, self.skeleton.nb_joints, 3, 3),
            true_hip_offset=root_positions.view(-1, 3))
        joint_rotations_mtx_global = joint_rotations_mtx_global.view(batch_size, n_frames, -1, 3, 3)
        joint_positions_global = joint_positions_global.view(batch_size, n_frames, -1, 3)

        joint_rotations_global = matrix_to_quaternion(joint_rotations_mtx_global)
        joint_rotations_ortho6d_global = matrix_to_rotation_6d(joint_rotations_mtx_global)

        # Add to the input dictionary
        input_data['joint_rotations_mtx'] = joint_rotations_mtx
        input_data['joint_rotations_ortho6d'] = joint_rotations_ortho6d
        input_data['joint_rotations_global'] = joint_rotations_global
        input_data['joint_rotations_mtx_global'] = joint_rotations_mtx_global
        input_data['joint_positions_global'] = joint_positions_global
        input_data['joint_rotations_ortho6d_global'] = joint_rotations_ortho6d_global

        return input_data

    def prepare_inputs(self, input_data):
        # Merge pas and future and expand
        root_positions = torch.cat([input_data['past_root_positions'], input_data['future_root_positions']], dim=1)
        joint_rotations = torch.cat([input_data['past_local_quats'], input_data['future_local_quats']], dim=1)
        input_frame_indices = torch.cat([input_data['past_frame_indices'], input_data['future_frame_indices']], dim=0)

        new_input_dict = {
            'root_positions': root_positions,
            'joint_rotations': joint_rotations,
            'input_frame_indices': input_frame_indices,
            'target_frame_indices': input_data['output_frame_indices'],
            'past_frame_indices': input_data['past_frame_indices'],
            'future_frame_indices': input_data['future_frame_indices'],
        }
        return self.expand_input(new_input_dict)

    def expand_predictions(self, predictions):
        # Expand predictions in other representations:
        batch_size, n_frames = predictions['joint_positions_global'].shape[:2]

        joint_positions_global = predictions['joint_positions_global']
        joint_rotations_ortho6d = predictions['joint_rotations_ortho6d']

        joint_rotations_mtx = rotation_6d_to_matrix(joint_rotations_ortho6d.view(-1, 6))
        joint_rotations = matrix_to_quaternion(joint_rotations_mtx)

        joint_positions_fk, joint_rotations_mtx_global = self.skeleton.forward(
            joint_rotations_mtx.view(-1, self.skeleton.nb_joints, 3, 3),
            true_hip_offset=joint_positions_global[..., self.root_idx, :].reshape(-1, 3)
        )
        joint_positions_fk = joint_positions_fk.view(batch_size, n_frames, self.skeleton.nb_joints, 3)
        joint_rotations_global = matrix_to_quaternion(joint_rotations_mtx_global.view(-1, 3, 3))

        predictions = {
            "joint_positions_global": joint_positions_fk,
            "root_position": joint_positions_global[..., self.root_idx, :],
            "joint_rotations_ortho6d": joint_rotations_ortho6d.view(batch_size, n_frames, self.skeleton.nb_joints,
                                                                    6),
            "joint_rotations_mtx": joint_rotations_mtx.view(batch_size, n_frames, self.skeleton.nb_joints, 3, 3),
            "joint_rotations": joint_rotations.view(batch_size, n_frames, self.skeleton.nb_joints, 4),
            "joint_rotations_mtx_global": joint_rotations_mtx_global.view(batch_size, n_frames,
                                                                          self.skeleton.nb_joints, 3, 3),
            "joint_rotations_global": joint_rotations_global.view(batch_size, n_frames, self.skeleton.nb_joints, 4),
        }

        return predictions

    def shared_step(self, batch):
        past_frames, future_frames, target_frames = self.get_data_from_batch(batch, self.frame_sampler)
        
        target_data, predicted = self.forward_wrapped(past_frames, future_frames, target_frames)
        
        past_frames = self.expand_input(past_frames)
        future_frames = self.expand_input(future_frames)
        target_frames = self.expand_input(target_frames)
        
        # Target positions
        target_positions = target_data['joint_positions_global']
        predicted_positions = predicted['joint_positions_global']  

        # Get additional data for the reconstruction losses
        src_time = torch.cat([past_frames['frame_indices'], future_frames['frame_indices']], dim=0)
        tgt_time = target_frames["frame_indices"]

        context_positions_global = torch.cat([past_frames['joint_positions_global'], 
                                              future_frames['joint_positions_global']], dim=1)
        context_rotations_global = torch.cat([past_frames['joint_rotations_global'], 
                                              future_frames['joint_rotations_global']], dim=1)

        past_ref = past_frames['joint_rotations_global'][:, -1:, ...]
        target_joint_rotations_global = target_data['joint_rotations_global']
        predicted_joint_rotations_global = predicted['joint_rotations_global'] 

        predicted_joint_rotations_global = torch.cat([past_ref, predicted_joint_rotations_global], dim=1)
        predicted_joint_rotations_global = remove_quat_discontinuities(predicted_joint_rotations_global)
        predicted_joint_rotations_global = predicted_joint_rotations_global[:, 1:, ...]

        target_joint_rotations_global = torch.cat([past_ref, target_joint_rotations_global], dim=1)
        target_joint_rotations_global = remove_quat_discontinuities(target_joint_rotations_global)
        target_joint_rotations_global = target_joint_rotations_global[:, 1:, ...]

        rot_loss_quat_global = self.l1_quaternion_loss(predicted_joint_rotations_global.contiguous()[:, tgt_time], 
                                                       target_joint_rotations_global.contiguous())

        pos_loss_l1 = self.l1_position_loss(predicted_positions.contiguous()[:, tgt_time], 
                                            target_positions.contiguous())

        # Compute the reconstruction losses
        pos_loss_l1_reconstruction = self.l1_position_loss(predicted_positions.contiguous()[:, src_time], 
                                                           context_positions_global.contiguous())
        rot_loss_quat_global_reconstruction = self.l1_quaternion_loss(predicted_joint_rotations_global.contiguous()[:, src_time], 
                                                                      context_rotations_global.contiguous())

        total_loss = self.hparams.quat_loss_scale * rot_loss_quat_global + \
                     self.hparams.pos_loss_l1_scale * pos_loss_l1 + \
                     self.hparams.reconstruction_scale * self.hparams.pos_loss_l1_scale * pos_loss_l1_reconstruction + \
                     self.hparams.reconstruction_scale * self.hparams.quat_loss_scale * rot_loss_quat_global_reconstruction
        
        return {
            "total": total_loss,
            "rot_loss_quat_global": rot_loss_quat_global,
            "pos_loss_l1": pos_loss_l1,
            "pos_loss_l1_reconstruction": pos_loss_l1_reconstruction,
            "rot_loss_quat_global_reconstruction": rot_loss_quat_global_reconstruction,
        }
    
    def configure_optimizers(self):
        # TODO: this is a hack can we pass the real None through config?
        if self.hparams.optimizer != 'None' and self.hparams.optimizer is not None:
            optimizer = instantiate(self.hparams.optimizer, params=self.parameters())
            scheduler = instantiate(self.hparams.scheduler, optimizer=optimizer)
            return {'optimizer': optimizer, 'lr_scheduler': scheduler}
    
    def test_step(self, batch, batch_idx):
        # TODO: clean this up, test metrics should only be defined at the task level and not depend on a specific model
        past_frames, future_frames, target_frames = self.get_data_from_batch(batch, frame_sampler=self.frame_sampler)
        target_data, predicted_data = self.forward_wrapped(past_frames, future_frames, target_frames)
        # TODO: clean this up as well, it will not work if we add other metrics on other outputs
        # Target should include all frames that are predicted
        tgt_time = target_frames["frame_indices"]
        predicted_data["joint_rotations"] = predicted_data["joint_rotations"][:, tgt_time]
        self.update_test_metrics(predicted_data, target_data)

    def export(self, filepath: str, **kwargs):
        dirpath = os.path.dirname(filepath)
        if not os.path.exists(dirpath):
            try:
                os.makedirs(dirpath)
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

        dummy_input = self.get_dummy_input()
        dynamic_axes = self.get_dynamic_axes()
        metadata = self.get_metadata()
        metadata_json = {"json": json.dumps(metadata)}
        export_named_model_to_onnx(self, dummy_input, filepath, metadata=metadata_json, dynamic_axes=dynamic_axes, verbose=True, **kwargs)

    def get_subsequence_from_batch(self, batch: Dict[str, torch.Tensor], frame_indices: List):
        """
        Batch is assumed to consist of tensors in the format
        B, T, F, 3 - positions
        B, T, F, 4 - quaternions
        B, T, F, 3, 3 - rotation matrices
        """
        subsequence = dict()
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                subsequence[k] = v[:, frame_indices, ...]
        return subsequence

    def get_batch_frame_indices(self, batch_length, training=True):
        assert batch_length >= 3
        past_indices, future_indices, output_indices = self.frame_sampler(list(range(batch_length)))
        return past_indices, future_indices, output_indices

    def get_data_from_batch(self, batch: Dict[str, torch.Tensor], frame_sampler: Any):
        past_indices, future_indices, output_indices = frame_sampler(list(range(batch['sequence_length'])))

        joint_positions_global = batch['joint_positions']  # Global positions
        root_positions = joint_positions_global[:, :, self.root_idx, :]  # Global root positions
        joint_rotations = batch['joint_rotations']  # Local quats

        # This is the minimal information this task needs:
        batch = {'root_positions': root_positions, 'joint_rotations': joint_rotations}

        past_indices = torch.Tensor(past_indices).to(dtype=torch.int64, device=self.device)
        future_indices = torch.Tensor(future_indices).to(dtype=torch.int64, device=self.device)
        output_indices = torch.Tensor(output_indices).to(dtype=torch.int64, device=self.device)

        past_frames = self.get_subsequence_from_batch(batch, past_indices)
        future_frames = self.get_subsequence_from_batch(batch, future_indices)
        target_frames = self.get_subsequence_from_batch(batch, output_indices)

        # Prevent out-of-range indices, start at idx 0
        start_frame = past_indices[0].clone()
        past_frames['frame_indices'] = past_indices - start_frame
        future_frames['frame_indices'] = future_indices - start_frame
        target_frames['frame_indices'] = output_indices - start_frame

        return past_frames, future_frames, target_frames

    def get_dummy_input(self):
        return {
            "past_frame_indices": torch.zeros((10), dtype=torch.int64),
            "future_frame_indices": torch.zeros((1), dtype=torch.int64),
            "output_frame_indices": torch.zeros((5), dtype=torch.int64),
            "past_root_positions": torch.randn((1, 10, 3)),
            "future_root_positions": torch.randn((1, 1, 3)),
            "past_local_quats": torch.randn((1, 10, self.skeleton.nb_joints, 4)),
            "future_local_quats": torch.randn((1, 1, self.skeleton.nb_joints, 4)),
        }

    def get_dynamic_axes(self):
        return {
            'past_frame_indices': {0: 'number_of_past_frames'},
            'future_frame_indices': {0: 'number_of_future_frames'},
            'output_frame_indices': {0: 'number_of_output_frames'},
            'past_root_positions': {1: 'number_of_past_frames'},
            'future_root_positions': {1: 'number_of_future_frames'},
            'past_local_quats': {1: 'number_of_past_frames'},
            'future_local_quats': {1: 'number_of_future_frames'},
            'joint_positions_global': {1: 'number_of_output_frames'},
            'joint_rotations_ortho6d': {1: 'number_of_output_frames'},
        }

    def get_metadata(self):
        metadata = {"model": type(self).__name__}
        # IMPORTANT: this metadata structure MUST match the one on the C# side
        model_params = {
            # Training regime:
            "training_min_past_frames": self.hparams.frame_sampler.min_past_context,
            "training_max_past_frames": self.hparams.frame_sampler.max_past_context,
            "training_min_future_frames": 1,
            "training_max_future_frames": self.hparams.frame_sampler.max_future_context,
            "training_min_output_frames": self.hparams.frame_sampler.min_middle_frames,
            "training_max_output_frames": self.hparams.frame_sampler.max_middle_frames,

            # Hard constraints (0 => infinity)
            # If hyper-params have an impact on these hard constraints, logic should be added here
            "allowed_min_past_frames": 1,
            "allowed_max_past_frames": 0,
            "allowed_min_future_frames": 1,
            "allowed_max_future_frames": 0,
            "allowed_min_output_frames": 1,
            "allowed_max_output_frames": 0,

            "transpose_ortho6d": True  # pytorch3d uses rows instead of columns
        }
        metadata["model_params"] = model_params
        metadata["skeleton"] = self.skeleton.full_hierarchy
        return metadata

    @staticmethod
    def get_metrics():
        return {}

    def test_step_end(self, *args, **kwargs):
        super().test_step_end(args, kwargs)

    def update_test_metrics(self, predicted, target):
        target_joint_rotations = target["joint_rotations"]
        predicted_joint_rotations = predicted["joint_rotations"]

    def on_train_epoch_start(self) -> None:
        try:
            dataset = self.trainer.train_dataloader.dataset
            dataset.set_epoch(self.current_epoch)
        except Exception:
            pass
        return super().on_train_epoch_start()

    def log_train_losses(self, losses: Dict[str, Any], prefix: str = ""):
        for k, v in losses.items():
            if v is not None:
                self.log("train/" + prefix + k, v, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

    def log_validation_losses(self, losses: Dict[str, Any], prefix: str = ""):
        for k, v in losses.items():
            if v is not None:
                self.log("validation/" + prefix + k, v, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

    def get_dummy_output(self):
        return {}

    def get_joint_indices(self, joint_names: Union[str, List[str]]):
        if isinstance(joint_names, str):
            return self.skeleton.bone_indexes[joint_names]
        else:
            return [self.skeleton.bone_indexes[name] for name in joint_names]
