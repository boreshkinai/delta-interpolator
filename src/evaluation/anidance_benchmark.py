import torch
import numpy as np
import logging
import copy

from src.data.datasets import SplitFileDatabaseLoader
from src.data.typed_table import TypedColumnSequenceDataset
from src.evaluation.l2p_error import L2P
from src.data.frame_sampler import MiddleFramesRemover
from src.data.batched_sequence_dataset import AnidanceSequenceDataset


class AnidanceBenchmarkEvaluator:
    def __init__(self, 
                 datasets_path: str = 'datasets/anidance', 
                 benchmark: str = 'anidance', 
                 device: str = 'cpu', 
                 verbose: bool = False):
        
        self.datasets_path = datasets_path
        self.benchmark = benchmark
        self.device = device
        self.verbose = verbose
        self.n_past = 1
        self.n_future = 1
        
        full_dataset = TypedColumnSequenceDataset('datasets/anidance/dances')
        full_dataset.remove_short_sequences(128)
        full_dataset.format_as_sliding_windows(128, 64)
        
        split = SplitFileDatabaseLoader(self.datasets_path).split_file_of(self.datasets_path)

        training_dataset = TypedColumnSequenceDataset(split, subset="Training")
        training_dataset.remove_short_sequences(128)
        training_dataset.format_as_sliding_windows(128, 64)
        
        validation_dataset = TypedColumnSequenceDataset(split, subset="Validation")
        validation_dataset.remove_short_sequences(128)
        validation_dataset.format_as_sliding_windows(128, 64)

        self.training_dataset = AnidanceSequenceDataset(source=training_dataset,
                                           batch_size=64,  shuffle=False,  drop_last=False,
                                           seed=0, min_length=128, max_length=128)

        self.validation_dataset = AnidanceSequenceDataset(source=validation_dataset,
                                           batch_size=64,  shuffle=False,  drop_last=False,
                                           seed=0, min_length=128, max_length=128)

       
        self.training_dataset.compute_stats()  

        self.l2p = L2P(self.training_dataset.x_mean.to(self.device), 
                       self.training_dataset.x_std.to(self.device))
        
        self.frame_samplers = dict()
        for n_trans in [5, 15, 30]:
            self.frame_samplers[n_trans] = MiddleFramesRemover(past_context=self.n_past, 
                                                               future_context=self.n_future, 
                                                               middle_frames=n_trans-1)
            
                    
    @torch.no_grad()
    def evaluate(self, model):
        model.eval()
        metrics = {}
        for n_trans in [5, 15, 30]:
            self.l2p.reset()
            for i in range(len(self.validation_dataset)):
                b = self.validation_dataset[i]
                for k, v in b.items():
                    if isinstance(v, torch.Tensor):
                        b[k] = v.to(model.device)
                    
                past_frames, future_frames, target_frames = model.get_data_from_batch(b, frame_sampler=self.frame_samplers[n_trans])

                target_data, predicted = model.forward_wrapped(past_frames, future_frames, target_frames)
                
                tgt_indices = target_frames['frame_indices']
                positions_preds = predicted['joint_positions']
                if type(positions_preds) ==  tuple:
                    positions_preds = positions_preds[0]
                positions_preds = positions_preds[:, tgt_indices]
                positions_target = target_data['joint_positions']
               
                self.l2p.update(preds = positions_preds.view(*positions_preds.shape[:2], 
                                                             np.prod(positions_preds.shape[-2:])).to(self.device), 
                                target = positions_target.view(*positions_target.shape[:2], 
                                                               np.prod(positions_target.shape[-2:])).to(self.device))
                
            
            metrics[f"L2P@{n_trans}"] = np.round(self.l2p.compute().numpy(), 3)
        
        if self.verbose:
            logging.info("================== ANIDANCE BENCHMARK ======================")
            for k, v in metrics.items():
                logging.info("%s: %f" % (k, v))
            logging.info("============================================================")
            
        return metrics
