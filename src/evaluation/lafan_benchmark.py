import torch
import numpy as np
import logging

from src.data.datasets import SplitFileDatabaseLoader
from src.data.typed_table import TypedColumnSequenceDataset
from src.geometry.skeleton import Skeleton
from src.data.batched_sequence_dataset import LafanSequenceDataset
from src.evaluation.l2q_error import L2Q
from src.evaluation.l2p_error import L2P
from src.evaluation.npss_error import NPSS
from src.data.frame_sampler import MiddleFramesRemover
from src.data.augmentation import BatchRemoveQuatDiscontinuities, BatchYRotateOnFrame, \
    BatchCenterXZ

from src.geometry.quaternions import (
    remove_quat_discontinuities
)

VALID_BENCHMARKS = {"lafan": "deeppose_lafan_v1_fps30"}


class LafanBenchmarkEvaluator:
    def __init__(self, 
                 datasets_path: str = '../datasets', 
                 benchmark: str = 'lafan', 
                 device: str = 'cpu', 
                 verbose: bool = False):
        
        self.datasets_path = datasets_path
        self.benchmark = benchmark
        self.device = device
        self.verbose = verbose
        self.n_past = 10
        self.n_future = 1
        self.n_trans = [5, 15, 30]
        
        dataset = VALID_BENCHMARKS[self.benchmark]
        
        split = SplitFileDatabaseLoader(self.datasets_path).pull(dataset)
        lafan_train_raw = TypedColumnSequenceDataset(split, subset="Training")
        lafan_val_raw = TypedColumnSequenceDataset(split, subset="Validation")
        
        lafan_train_raw.remove_short_sequences(50)
        lafan_train_raw.format_as_sliding_windows(50, 20)

        lafan_val_raw.remove_short_sequences(65)
        lafan_val_raw.format_as_sliding_windows(65, 40)
        
        skeleton_data = lafan_train_raw.config["skeleton"]
        skeleton = Skeleton(skeleton_data)
        skeleton.remove_joints(['LeftToeEnd', 'RightToeEnd', 'LeftHandEnd', 'RightHandEnd', 'HeadEnd'])
        self.skeleton = skeleton
        
        self.training_dataset = LafanSequenceDataset(source=lafan_train_raw, skeleton=skeleton,
                                   batch_size=64,  shuffle=False,  drop_last=False,
                                   seed=0, min_length=50, max_length=50)

        self.validation_dataset = LafanSequenceDataset(source=lafan_val_raw, skeleton=skeleton,
                                           batch_size=64,  shuffle=False,  drop_last=False,
                                           seed=0, min_length=65, max_length=65)

        transforms = []
        transforms.append(BatchCenterXZ())
        transforms.append(BatchYRotateOnFrame(skeleton, rotation_frame=9))
        transforms.append(BatchRemoveQuatDiscontinuities())

        self.training_dataset.add_transforms(transforms)
        self.validation_dataset.add_transforms(transforms)

        self.training_dataset.compute_stats()  
        
        self.l2p = L2P(self.training_dataset.x_mean.to(self.device), 
                       self.training_dataset.x_std.to(self.device))
        self.l2q = L2Q()
        self.npss = NPSS()
        
        self.l2p_key = L2P(self.training_dataset.x_mean.to(self.device), 
                           self.training_dataset.x_std.to(self.device))
        self.l2q_key = L2Q()
        self.npss_key = NPSS()
        
        self.frame_samplers = dict()
        for n_trans in self.n_trans:
            self.frame_samplers[n_trans] = MiddleFramesRemover(past_context=self.n_past, 
                                                               future_context=self.n_future, 
                                                               middle_frames=n_trans)
        
    @torch.no_grad()
    def evaluate(self, model):
        model.eval()
        metrics = {}
        for n_trans in self.n_trans:
            self.l2p.reset()
            self.l2q.reset()
            self.npss.reset()
            
            self.l2p_key.reset()
            self.l2q_key.reset()
            self.npss_key.reset()
            
            for i in range(len(self.validation_dataset)):
                b = self.validation_dataset[i]
                for k, v in b.items():
                    if isinstance(v, torch.Tensor):
                        b[k] = v.to(model.device)
                    
                past_frames, future_frames, target_frames = model.get_data_from_batch(b, frame_sampler=self.frame_samplers[n_trans])
                
                target_data, predicted = model.forward_wrapped(past_frames, future_frames, target_frames)
                
                past_frames = model.expand_input(past_frames)
                future_frames = model.expand_input(future_frames)
                target_frames = model.expand_input(target_frames)

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
                expanded_input_data = model.prepare_inputs(input_data)
                
                tgt_indices = target_frames['frame_indices']
                positions_preds = predicted['joint_positions_global'][:, tgt_indices]
                positions_target = target_data['joint_positions_global']
                
                self.l2p.update(preds = positions_preds.view(*positions_preds.shape[:2], 
                                                             np.prod(positions_preds.shape[-2:])).to(self.device), 
                                target = positions_target.view(*positions_target.shape[:2], 
                                                               np.prod(positions_target.shape[-2:])).to(self.device))
                
                rotations_preds = predicted['joint_rotations_global'][:, tgt_indices]
                rotations_target = target_data['joint_rotations_global']
                
                # Implement the quaternion continuity correction
                reference_quat = expanded_input_data['joint_rotations_global'][:, self.n_past-1:self.n_past, ...]
                rotations_preds = remove_quat_discontinuities(torch.cat([reference_quat, rotations_preds], dim=1))
                rotations_target = remove_quat_discontinuities(torch.cat([reference_quat, rotations_target], dim=1))
                rotations_preds = rotations_preds[:, 1:, ...]
                rotations_target = rotations_target[:, 1:, ...]
                
                self.l2q.update(preds=rotations_preds.to(self.device), 
                                target=rotations_target.to(self.device))

                self.npss.update(preds=rotations_preds.to(self.device),
                                 target=rotations_target.to(self.device))
                
                # Evaluate metrics on the key frames
                src_indices = torch.cat([past_frames['frame_indices'], future_frames['frame_indices']], dim=0)
                positions_preds = predicted['joint_positions_global'][:, src_indices]
                context_positions_global = torch.cat([past_frames['joint_positions_global'], 
                                                      future_frames['joint_positions_global']], dim=1)
        
                
                self.l2p_key.update(preds = positions_preds.view(*positions_preds.shape[:2],
                                                                 np.prod(positions_preds.shape[-2:])).to(self.device), 
                                    target = context_positions_global.view(
                                        *context_positions_global.shape[:2], 
                                        np.prod(context_positions_global.shape[-2:])).to(self.device)
                                                                           )
                
                rotations_preds = predicted['joint_rotations_global']
                rotations_target = torch.cat([past_frames['joint_rotations_global'], 
                                              target_frames['joint_rotations_global'], 
                                              future_frames['joint_rotations_global']], dim=1)
                reference_quat = expanded_input_data['joint_rotations_global'][:, :1]
                
                rotations_preds = remove_quat_discontinuities(torch.cat([reference_quat, rotations_preds], dim=1))
                rotations_target = remove_quat_discontinuities(torch.cat([reference_quat, rotations_target], dim=1))
                rotations_preds = rotations_preds[:, 1:][:, src_indices]
                rotations_target = rotations_target[:, 1:][:, src_indices]
                
                self.l2q_key.update(preds=rotations_preds.to(self.device), 
                                    target=rotations_target.to(self.device))

                self.npss_key.update(preds=rotations_preds.to(self.device),
                                     target=rotations_target.to(self.device))
                
                        
            metrics[f"L2P@{n_trans}"] = self.l2p.compute().numpy().item()
            metrics[f"L2Q@{n_trans}"] = self.l2q.compute().numpy().item()
            metrics[f"NPSS@{n_trans}"] = self.npss.compute().numpy().item()
            
            metrics[f"L2P_KEY@{n_trans}"] = self.l2p_key.compute().numpy().item()
            metrics[f"L2Q_KEY@{n_trans}"] = self.l2q_key.compute().numpy().item()
            metrics[f"NPSS_KEY@{n_trans}"] = self.npss_key.compute().numpy().item()
                
        if self.verbose:
            logging.info("=================== LAFAN BENCHMARK ========================")
            for k, v in metrics.items():
                logging.info("%s: %f" % (k, np.round(v, 3)))
            logging.info("============================================================")
            
        return metrics


