import pytorch_lightning as pl
from torch.utils.data import DataLoader
from src.geometry.skeleton import Skeleton
from src.data.augmentation import BatchRemoveQuatDiscontinuities, BatchYRotateOnFrame, \
    BatchCenterXZ, BatchRotate, BatchMirror

import hydra
from hydra.utils import instantiate
from dataclasses import dataclass
from typing import Any
from src.data.typed_table import TypedColumnSequenceDataset
from src.utils.python import get_full_class_reference
from src.data.datasets import DatasetLoader


# custom collate function for batched dataset
# Note: would be nicer as lambdas, but lambdas cannot be pickled and will thus break distributed training
def _batched_collate(batch):
    return batch[0]


class SequenceDataModule(pl.LightningDataModule):
    def __init__(self, backbone: Any, path: str, name: str, batch_size: int, num_workers: int = 0,
                 mirror: bool = True, rotate: bool = True, centerXZ: bool = True,
                 y_rotate_on_frame: int = -1, remove_quat_discontinuities: bool = True,
                 augment_training: bool = True, augment_validation: bool = False,
                 use_sliding_windows: bool = True, 
                 sequence_offset_train: int = 20, sequence_offset_validation: int = 40,
                 min_sequence_length_train: int = 20, max_sequence_length_train: int = 65,
                 sequence_length_validation: int = 30):
        super().__init__()

        self.path = hydra.utils.to_absolute_path(path)
        self.name = name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.skeleton = None
        self.mirror = mirror
        self.rotate = rotate
        self.centerXZ = centerXZ
        self.y_rotate_on_frame = y_rotate_on_frame
        self.remove_quat_discontinuities = remove_quat_discontinuities
        self.augment_training = augment_training
        self.augment_validation = augment_validation
        self.use_sliding_windows = use_sliding_windows
        self.sequence_offset_train = sequence_offset_train
        self.sequence_offset_validation = sequence_offset_validation
        self.min_sequence_length_train = min_sequence_length_train
        self.max_sequence_length_train = max_sequence_length_train
        self.sequence_length_validation = sequence_length_validation

        self.batched_dataset_backbone = backbone
        self.transforms = []

    def prepare_data(self):
        # download dataset
        dataset_loader = DatasetLoader(self.path)
        dataset_loader.pull(self.name)

    def get_skeleton(self) -> Skeleton:
        if self.skeleton is None:
            dataset_settings = DatasetLoader(self.path).get_settings(self.name)
            assert "skeleton" in dataset_settings, "No skeleton data could be found in dataset settings"
            self.skeleton = Skeleton(dataset_settings["skeleton"])
            self.skeleton.remove_joints(['LeftToeEnd', 'RightToeEnd', 'LeftHandEnd', 'RightHandEnd', 'HeadEnd'])
            return self.skeleton
        else:
            return self.skeleton

    def setup(self, stage=None):
        # retrieve train / validation split
        self.split = DatasetLoader(self.path).get_split(self.name)

        # retrieve skeleton
        self.get_skeleton()
        
        # load datasets
        training_dataset, validation_dataset = TypedColumnSequenceDataset.FromSplit(self.split)
        
        training_dataset.remove_short_sequences(self.max_sequence_length_train)
        training_dataset.format_as_sliding_windows(self.max_sequence_length_train, self.sequence_offset_train)
        
        validation_dataset.remove_short_sequences(self.sequence_length_validation)
        validation_dataset.format_as_sliding_windows(self.sequence_length_validation, self.sequence_offset_validation)

        # Wrap datasets into their batched version
        #   We need to do that as a wrapped dataset as Lightning doesn't support
        #   well custom samplers with distributed training
        #   As a consequence of that, we skipp collapsing in the data loader
        # Give the length information to the batched dataset as we want to all sequences in a batch to be of same length
        self.training_dataset = instantiate(self.batched_dataset_backbone, training_dataset,
                                                     skeleton=self.skeleton,
                                                     batch_size=self.batch_size,
                                                     min_length=self.min_sequence_length_train,
                                                     max_length=self.max_sequence_length_train,
                                                     shuffle=True,
                                                     drop_last=True)

        # TODO make sure that even when NOT augmenting validation, we perform the required processing:
        #  centering, Y rotating, removing quat discontinuities.
        self.pre_transforms = []
        self.augmentation = []
        self.post_transforms = []

        # Pre-processing
        if self.centerXZ:
            self.pre_transforms.append(BatchCenterXZ())
        if self.y_rotate_on_frame >= 0:
            self.pre_transforms.append(BatchYRotateOnFrame(self.skeleton, rotation_frame=self.y_rotate_on_frame))
        # Augmentation
        if self.mirror:
            self.augmentation.append(BatchMirror(self.skeleton, mirror_prob=0.5))
        if self.rotate:
            self.augmentation.append(BatchRotate(self.skeleton))
        # Post-processing
        if self.remove_quat_discontinuities:
            self.post_transforms.append(BatchRemoveQuatDiscontinuities())

        self.training_dataset.add_transforms(self.pre_transforms)
        if self.augment_training:
            self.training_dataset.add_transforms(self.augmentation)
        self.training_dataset.add_transforms(self.post_transforms)

        self.validation_dataset = instantiate(self.batched_dataset_backbone, validation_dataset,
                                                       skeleton=self.skeleton,
                                                       batch_size=self.batch_size,
                                                       min_length=self.sequence_length_validation,
                                                       max_length=self.sequence_length_validation,
                                                       shuffle=False,  
                                                       drop_last=False)

        self.validation_dataset.add_transforms(self.pre_transforms)
        if self.augment_validation:
            self.validation_dataset.add_transforms(self.augmentation)
        self.validation_dataset.add_transforms(self.post_transforms)

    def train_dataloader(self):
        return DataLoader(self.training_dataset, batch_size=1, shuffle=True, num_workers=self.num_workers,
                          collate_fn=_batched_collate)

    def val_dataloader(self):
        return DataLoader(self.validation_dataset, batch_size=1, shuffle=False, num_workers=self.num_workers,
                          collate_fn=_batched_collate)

    def test_dataloader(self):
        return self.val_dataloader()


@dataclass
class SequenceDataModuleOptions:
    _target_: str = get_full_class_reference(SequenceDataModule)
    backbone: Any = None
    path: str = "./datasets"
    name: str = "deeppose_master_v1_fps60"
    batch_size: int = 32
    num_workers: int = 0
    sequence_offset_train: int = 20
    min_sequence_length_train: int = 20
    max_sequence_length_train: int = 50
    use_sliding_windows: bool = True
    sequence_offset_validation: int = 40
    sequence_length_validation: int = 65
    mirror: bool = True
    rotate: bool = True
    augment_training: bool = True
    augment_validation: bool = False
    centerXZ: bool = True
    y_rotate_on_frame: int = -1
    remove_quat_discontinuities: bool = True
