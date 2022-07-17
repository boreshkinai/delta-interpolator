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
from src.data.sequence_module import AlternateSequenceDataModuleOptions

from src.utils.model_factory import ModelFactory
from src.utils.options import BaseOptions

from omegaconf import DictConfig

from src.evaluation.anidance_benchmark import AnidanceBenchmarkEvaluator


@dataclass
class AnidanceInBetweenModelOptions(BaseOptions):
    frame_sampler: Any = RandomMiddleFramesRemoverOptions(min_past_context=10,
                                                          max_past_context=10,
                                                          max_future_context=1,
                                                          min_middle_frames=5,
                                                          max_middle_frames=39)
    dataset:  AlternateSequenceDataModuleOptions = AlternateSequenceDataModuleOptions()
    optimizer: Any = None
    scheduler: Any = None
    backbone: Any = None
    pos_loss_l1_scale: float = 10.0
    reconstruction_scale: float = 1.0
    datasets_path: str = "./datasets/anidance/dances"
    benchmark: str = "anidance"

        
@ModelFactory.register(AnidanceInBetweenModelOptions, schema_name="AnidanceInBetween")
class AnidanceInBetweenModel(LightningModule):

    def __init__(self, opts: AnidanceInBetweenModelOptions):
        super().__init__()

        assert isinstance(opts,
                          DictConfig), f"opt constructor argument must be of type DictConfig but got {type(opts)} instead."

        self.save_hyperparameters(opts)

        self.frame_sampler = instantiate(opts.frame_sampler)
        self.net = instantiate(opts.backbone, nb_joints=24)
        
        self.evaluator = None
        if opts.benchmark != 'None':
            self.evaluator = AnidanceBenchmarkEvaluator(datasets_path=hydra.utils.to_absolute_path(opts.datasets_path),
                                                     benchmark=opts.benchmark, 
                                                     device='cpu', 
                                                     verbose=True)

    def forward(self, input_data):
        # print(input_data.keys())
        # print('here1', type(input_data['past_joint_positions']))
        # print('here1', input_data['past_joint_positions'].shape)
        # print('here1', input_data['joint_positions'].shape)
        # print('here1', input_data['future_joint_positions'].shape)
        input_data = self.prepare_inputs(input_data)
        joint_positions = self.net(input_data=input_data)
        return {
            "joint_positions": joint_positions,
        }

    def forward_wrapped(self, past_frames, future_frames, target_frames):
        """
        Useful to have to avoid having external object, such as the evaluator recreate this chunk of the shared step
        """
        target_data = target_frames

        # print('target_frames', target_frames.keys())

        # Must match what is defined in the task.
        # print('Past positions', past_frames['joint_positions'].shape)
        input_data = {
            "past_frame_indices": past_frames['frame_indices'],
            "future_frame_indices": future_frames['frame_indices'],
            "output_frame_indices": target_frames['frame_indices'],
            "past_joint_positions": past_frames['joint_positions'],
            "future_joint_positions": future_frames['joint_positions'],
        }
        
        predicted = self.forward(input_data=input_data)
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
    
    def prepare_inputs(self, input_data):
        # Merge past and future and expand
        # print(input_data.keys())
        # if "past_joint_positions" not in input_data.keys():
        #     print(input_data)
        #     print(input_data.keys())
        #     exit()
        # print('here3', type(input_data['past_joint_positions']))
        # print(input_data['past_joint_positions'])
        joint_positions = None
        input_frame_indices = None
        # print('Past positions', input_data['past_joint_positions'].shape)
        if type(input_data['past_joint_positions']) is dict:
            # print(input_data['past_joint_positions'].keys())
            # print(input_data['future_joint_positions'].keys())
            # print('input', input_data['future_joint_positions']['frame'].shape)
            # print('input', input_data['past_frame_indices']['joint_positions'].shape)
            # torch.cat([input_data['past_joint_positions']['joint_positions'], input_data['future_joint_positions']['joint_positions']], dim=1).shape
            joint_positions = torch.cat([input_data['past_joint_positions']['joint_positions'], input_data['future_joint_positions']['joint_positions']], dim=1)
            input_frame_indices = torch.cat([input_data['past_frame_indices'], input_data['future_frame_indices']], dim=0)
        else:
            joint_positions = torch.cat([input_data['past_joint_positions'], input_data['future_joint_positions']], dim=1)
            input_frame_indices = torch.cat([input_data['past_frame_indices'], input_data['future_frame_indices']], dim=0)

        new_input_dict = {
            'joint_positions': joint_positions,
            'input_frame_indices': input_frame_indices,
            'target_frame_indices': input_data['output_frame_indices'],
            'past_frame_indices': input_data['past_frame_indices'],
            'future_frame_indices': input_data['future_frame_indices'],
        }
        return new_input_dict

    # def expand_predictions(self, predictions):
    #     # Expand predictions in other representations:
    #     # print('pred', predictions)
    #     batch_size, n_frames = predictions['joint_positions'].shape[:2]

    #     joint_positions_global = predictions['joint_positions']

    #     predictions = {
    #         "joint_positions": predictions
    #     }

    #     return predictions
    def expand_input(self, input_data):
        # Add different data representations to the input
        joint_positions = input_data['joint_positions']
        batch_size, n_frames = joint_positions.shape[:2]

        # Add to the input dictionary
        input_data['joint_positions'] = joint_positions
        return input_data


    def shared_step(self, batch):
        past_frames, future_frames, target_frames = self.get_data_from_batch(batch, self.frame_sampler)
        
        target_data, predicted = self.forward_wrapped(past_frames, future_frames, target_frames)

        past_frames = self.expand_input(past_frames)
        future_frames = self.expand_input(future_frames)
        target_frames = self.expand_input(target_frames)
        
        # Target positions
        target_positions = target_data['joint_positions']
        # predicted_positions = predicted['joint_positions'][0]
        predicted_positions = predicted['joint_positions'][0]
        predicted_positions_2 = predicted['joint_positions'][1]

        # Get additional data for the reconstruction losses
        src_time = torch.cat([past_frames['frame_indices'], future_frames['frame_indices']], dim=0)
        tgt_time = target_frames["frame_indices"]

        context_positions_global = torch.cat([past_frames['joint_positions'], 
                                              future_frames['joint_positions']], dim=1)

        pos_loss_l1 = self.l1_position_loss(predicted_positions.contiguous()[:, tgt_time], 
                                            target_positions.contiguous())

        # Compute the reconstruction losses
        pos_loss_l1_reconstruction = self.l1_position_loss(predicted_positions.contiguous()[:, src_time], 
                                                           context_positions_global.contiguous())
        
        total_loss = self.hparams.pos_loss_l1_scale * pos_loss_l1 + \
                     self.hparams.reconstruction_scale * self.hparams.pos_loss_l1_scale * pos_loss_l1_reconstruction
        
        return {
            "total": total_loss,
            "pos_loss_l1": pos_loss_l1,
            "pos_loss_l1_reconstruction": pos_loss_l1_reconstruction,
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
        pass

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

        joint_positions = batch['joint_positions']  # Global positions

        # print(batch.keys())
        # This is the minimal information this task needs:
        batch = {'joint_positions': joint_positions}

        # print(batch.keys())
        # exit()
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

        # print(past_frames)

        return past_frames, future_frames, target_frames

    def get_dummy_input(self):
        return {
            "past_frame_indices": torch.zeros((10), dtype=torch.int64),
            "future_frame_indices": torch.zeros((1), dtype=torch.int64),
            "output_frame_indices": torch.zeros((5), dtype=torch.int64),
            "past_joint_positions": torch.randn((1, 10, 3 * 24)),
            "future_joint_positions": torch.randn((1, 1, 3 * 24)),
        }

    def get_dynamic_axes(self):
        return {
            'past_frame_indices': {0: 'number_of_past_frames'},
            'future_frame_indices': {0: 'number_of_future_frames'},
            'output_frame_indices': {0: 'number_of_output_frames'},
            'past_joint_positions': {1: 'number_of_past_frames'},
            'future_joint_positions': {1: 'number_of_future_frames'},
            'joint_positions_global': {1: 'number_of_output_frames'},
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

        }
        metadata["model_params"] = model_params
        return metadata

    @staticmethod
    def get_metrics():
        return {}

    def test_step_end(self, *args, **kwargs):
        super().test_step_end(args, kwargs)

    def update_test_metrics(self, predicted, target):
        target_joint_rotations = target["joint_positions"]
        predicted_joint_rotations = predicted["joint_positions"]

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
