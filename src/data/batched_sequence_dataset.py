from typing import List
import torch
import random
from torch.utils.data import Dataset
from src.data.typed_table import TypedColumnSequenceDataset
from src.geometry.skeleton import Skeleton


class BaseBatchedSequenceDataset(Dataset):
    """
    Base class for batched sequence data. Defines the interface the batched sequence objects must have to be used in the
    sequence_module.
    """

    def __init__(self, source: TypedColumnSequenceDataset,
                 skeleton: Skeleton = None,
                 batch_size: int = 32,
                 shuffle: bool = True,
                 drop_last: bool = True,
                 seed: int = 0,
                 min_length: int = 2,
                 max_length: int = 30):
        super().__init__()

        self.skeleton = skeleton
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.batch_indices = None
        self.seed = seed
        self.epoch = 0
        self.min_length = min_length
        self.max_length = max_length
        self.transforms = None

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self):
        raise NotImplementedError

    def add_transforms(self, transforms: List[torch.nn.Module]):
        if len(transforms) > 0:
            if self.transforms is None:
                self.transforms = transforms
            else:
                self.transforms = self.transforms + transforms


class BatchedSequenceDataset(BaseBatchedSequenceDataset):
    def __init__(self, source: TypedColumnSequenceDataset,
                 skeleton: Skeleton = None,
                 batch_size: int = 32,
                 shuffle: bool = True,
                 drop_last: bool = True,
                 seed: int = 0,
                 min_length: int = 2,
                 max_length: int = 30):
        super().__init__(source=source, skeleton=skeleton, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,
                         seed=seed, min_length=min_length, max_length=max_length)

        self.source = BaseSequenceDataset(source=source, skeleton=self.skeleton)
        self._shuffle()

    def __len__(self):
        if self.drop_last:
            return len(self.source) // self.batch_size
        else:
            return (len(self.source) + self.batch_size - 1) // self.batch_size

    def _shuffle(self):
        if self.shuffle:
            # seeding is needed to make it deterministic across processes and epochs
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            self.batch_indices = torch.randperm(len(self.source), generator=g).tolist()
        else:
            self.batch_indices = list(range(len(self.source)))

    def __getitem__(self, index):
        """
        As we don't clip the sequences here and giving a pd.Series of indices to the TypedColumnDatset object fails,
        we derived two different batching method, based on the underlying representation (fixed-length windows or not)
        :param index: minibatch index
        :return: tensor of fixed-sized data sequences.
        """
        start_idx = index * self.batch_size
        end_idx = min(len(self.source), start_idx + self.batch_size)
        batch_indices = self.batch_indices[start_idx:end_idx]

        batch = {}
        batch_length = random.randint(self.min_length, self.max_length)

        # When source dataset is not formatted as sliding windows,
        # we get variable length sequences one by one from the source dataset.
        # We then sample inside those sequences a fixed-length interval to creat our batch tensors.
        if not self.source.source.formatted_as_sliding_windows:
            for i in batch_indices:
                seq = self.source.__getitem__(i)
                seq_length = seq[list(seq.keys())[0]].shape[0]  # total sequence frames
                seq_start_idx = random.randint(0, seq_length - batch_length)  # random start frame
                for k, v in seq.items():
                    v = v[seq_start_idx: seq_start_idx + batch_length]

                    # Populate the batch dict with the same keys
                    batch.setdefault(k, []).append(v.unsqueeze(0))

            # Lists to tensors
            for k in batch.keys():
                batch[k] = torch.cat(batch[k])

        # When source data is formatted with sliding windows of fixed lengths, we can get the sequences in batch
        # by concatenating all windows indices (done at a lower level). This is much quicker.
        # However we still need to reshape, and sample a fixed-length range inside those tensors.
        # Still, for short sequences and large batches, this leads to a significant speedup.
        else:
            # Flat batch (B*T, J, D) to batched sequences (B, T, J, D)
            batch = self.source.__getitem__(batch_indices)
            for k, v in batch.items():
                v = v.view([len(batch_indices), -1] + list(v.shape)[1:])  # To (B, T, J, D)
                # TODO Here the random call will cause to break determinism in the multi-GPU setting
                #  --> We could have parts of the updates on different sequence lengths, which isn't bad, but:
                #  --> Multi-GPU training won't have the same curves as single-GPU.
                seq_start_idx = random.randint(0, v.shape[1] - batch_length)
                batch[k] = v[:, seq_start_idx:seq_start_idx + batch_length]  # Sample random sub-sequence

        # Batched data augmentation
        if self.transforms is not None:
            for transform_module in self.transforms:
                batch = transform_module.forward(batch)

        batch['sequence_length'] = batch_length

        return batch


class LafanSequenceDataset(BaseBatchedSequenceDataset):
    def __init__(self, source: TypedColumnSequenceDataset,
                 skeleton: Skeleton = None,
                 batch_size: int = 32,
                 shuffle: bool = True,
                 drop_last: bool = True,
                 seed: int = 0,
                 min_length: int = 2,
                 max_length: int = 30,
                 ):
        super().__init__(source=source, skeleton=skeleton, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,
                         seed=seed, min_length=min_length, max_length=max_length)

        self.source = source
        self.x_mean = None
        self.x_std = None
        self.transforms = None

        self.source_batched = BatchedSequenceDataset(self.source,
                                                     skeleton=self.skeleton,
                                                     batch_size=self.batch_size,
                                                     min_length=self.min_length,
                                                     max_length=self.max_length,
                                                     shuffle=self.shuffle,
                                                     drop_last=self.drop_last)

    def compute_stats(self):
        if self.x_mean is None:
            n_joints = len(self.skeleton.all_joints)
            positions = []
            for i in range(len(self)):
                b = self.__getitem__(i)
                positions.append(b['joint_positions'].view(-1, n_joints, 3))

            positions = torch.cat(positions, dim=0)

            self.x_mean = positions.mean(dim=0, keepdim=True).reshape(-1, 1, n_joints * 3)
            self.x_std = positions.std(dim=0, keepdim=True).reshape(-1, 1, n_joints * 3)

    def __len__(self):
        return self.source_batched.__len__()

    def __getitem__(self, index):
        batch = self.source_batched.__getitem__(index)

        # Apply transformations / augmentations
        if self.transforms is not None:
            for transform_module in self.transforms:
                batch = transform_module.forward(batch)

        return batch


class BaseSequenceDataset(Dataset):
    def __init__(self, source: TypedColumnSequenceDataset, skeleton: Skeleton):
        super().__init__()

        self.source = source
        self.skeleton = skeleton

        # get all joints in skeleton order
        self.all_joints = []
        for i in range(self.skeleton.nb_joints):
            self.all_joints.append(self.skeleton.index_bones[i])

        # compute feature indices
        self.joint_positions_idx = self.source.get_feature_indices(["BonePositions"], self.all_joints)
        self.joint_rotations_idx = self.source.get_feature_indices(["BoneRotations"], self.all_joints)

    def __getitem__(self, index):
        sequence = self.source.__getitem__(index)  # (T, D)

        # Note: we convert quaternions from x, y, z, w to w, x, y, z
        item = {
            "joint_positions": sequence[:, self.joint_positions_idx].view(-1, self.skeleton.nb_joints, 3),
            "joint_rotations": sequence[:, self.joint_rotations_idx].view(-1, self.skeleton.nb_joints, 4)[:, :, [3, 0, 1, 2]],
        }

        return item

    def __len__(self):
        return len(self.source)


class BaseAnidanceSequenceDataset(Dataset):
    def __init__(self, source: TypedColumnSequenceDataset, n_joints: int = 24):
        super().__init__()

        self.source = source
        self.n_joints = n_joints

        # compute feature indices
        self.joint_positions_idx = list(range(self.n_joints * 3))

    def __getitem__(self, index):
        sequence = self.source.__getitem__(index)  # (T, D)

        item = {
            "joint_positions": sequence[:, self.joint_positions_idx].view(-1, self.n_joints, 3),
        }

        return item

    def __len__(self):
        return len(self.source)



class BaseAnidanceBatchedSequenceDataset(Dataset):
    """
    Base class for batched sequence data. Defines the interface the batched sequence objects must have to be used in the
    sequence_module.
    """

    def __init__(self, source: TypedColumnSequenceDataset,
                 batch_size: int = 32,
                 shuffle: bool = True,
                 drop_last: bool = True,
                 seed: int = 0,
                 min_length: int = 2,
                 max_length: int = 30):
        super().__init__()

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.batch_indices = None
        self.seed = seed
        self.epoch = 0
        self.min_length = min_length
        self.max_length = max_length
        self.transforms = None

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self):
        raise NotImplementedError

    def add_transforms(self, transforms: List[torch.nn.Module]):
        if len(transforms) > 0:
            if self.transforms is None:
                self.transforms = transforms
            else:
                self.transforms = self.transforms + transforms


class BatchedAnidanceSequenceDataset(BaseAnidanceBatchedSequenceDataset):
    def __init__(self, source: TypedColumnSequenceDataset,
                 batch_size: int = 32,
                 shuffle: bool = True,
                 drop_last: bool = True,
                 seed: int = 0,
                 min_length: int = 2,
                 max_length: int = 30):
        super().__init__(source=source, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,
                         seed=seed, min_length=min_length, max_length=max_length)

        self.source = BaseAnidanceSequenceDataset(source=source)
        self._shuffle()

    def __len__(self):
        if self.drop_last:
            return len(self.source) // self.batch_size
        else:
            return (len(self.source) + self.batch_size - 1) // self.batch_size

    def _shuffle(self):
        if self.shuffle:
            # seeding is needed to make it deterministic across processes and epochs
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            self.batch_indices = torch.randperm(len(self.source), generator=g).tolist()
        else:
            self.batch_indices = list(range(len(self.source)))

    def __getitem__(self, index):
        """
        As we don't clip the sequences here and giving a pd.Series of indices to the TypedColumnDatset object fails,
        we derived two different batching method, based on the underlying representation (fixed-length windows or not)
        :param index: minibatch index
        :return: tensor of fixed-sized data sequences.
        """
        start_idx = index * self.batch_size
        end_idx = min(len(self.source), start_idx + self.batch_size)
        batch_indices = self.batch_indices[start_idx:end_idx]

        batch = {}
        batch_length = random.randint(self.min_length, self.max_length)

        # When source dataset is not formatted as sliding windows,
        # we get variable length sequences one by one from the source dataset.
        # We then sample inside those sequences a fixed-length interval to creat our batch tensors.
        if not self.source.source.formatted_as_sliding_windows:
            for i in batch_indices:
                seq = self.source.__getitem__(i)
                seq_length = seq[list(seq.keys())[0]].shape[0]  # total sequence frames
                seq_start_idx = random.randint(0, seq_length - batch_length)  # random start frame
                for k, v in seq.items():
                    v = v[seq_start_idx: seq_start_idx + batch_length]

                    # Populate the batch dict with the same keys
                    batch.setdefault(k, []).append(v.unsqueeze(0))

            # Lists to tensors
            for k in batch.keys():
                batch[k] = torch.cat(batch[k])

        # When source data is formatted with sliding windows of fixed lengths, we can get the sequences in batch
        # by concatenating all windows indices (done at a lower level). This is much quicker.
        # However we still need to reshape, and sample a fixed-length range inside those tensors.
        # Still, for short sequences and large batches, this leads to a significant speedup.
        else:
            # Flat batch (B*T, J, D) to batched sequences (B, T, J, D)
            batch = self.source.__getitem__(batch_indices)
            for k, v in batch.items():
                # print(v.shape)
                v = v.view([len(batch_indices), -1] + list(v.shape)[1:])  # To (B, T, J, D)
                # TODO Here the random call will cause to break determinism in the multi-GPU setting
                #  --> We could have parts of the updates on different sequence lengths, which isn't bad, but:
                #  --> Multi-GPU training won't have the same curves as single-GPU.
                seq_start_idx = random.randint(0, v.shape[1] - batch_length)
                batch[k] = v[:, seq_start_idx:seq_start_idx + batch_length]  # Sample random sub-sequence

        # Batched data augmentation
        if self.transforms is not None:
            for transform_module in self.transforms:
                batch = transform_module.forward(batch)

        batch['sequence_length'] = batch_length

        return batch


class AnidanceSequenceDataset(BaseAnidanceBatchedSequenceDataset):
    def __init__(self, source: TypedColumnSequenceDataset,
                 batch_size: int = 32,
                 shuffle: bool = True,
                 drop_last: bool = True,
                 seed: int = 0,
                 min_length: int = 2,
                 max_length: int = 30,
                 n_joints: int = 24
                 ):
        super().__init__(source=source, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,
                         seed=seed, min_length=min_length, max_length=max_length)

        self.source = source
        self.x_mean = None
        self.x_std = None
        self.n_joints = n_joints

        self.source_batched = BatchedAnidanceSequenceDataset(self.source,
                                                     batch_size=self.batch_size,
                                                     min_length=self.min_length,
                                                     max_length=self.max_length,
                                                     shuffle=self.shuffle,
                                                     drop_last=self.drop_last)

    def compute_stats(self):
        if self.x_mean is None:
            n_joints = self.n_joints
            positions = []
            for i in range(len(self)):
                b = self.__getitem__(i)
                positions.append(b['joint_positions'].reshape(-1, n_joints, 3))

            positions = torch.cat(positions, dim=0)

            self.x_mean = positions.mean(dim=0, keepdim=True).reshape(-1, 1, n_joints * 3)
            self.x_std = positions.std(dim=0, keepdim=True).reshape(-1, 1, n_joints * 3)

    def __len__(self):
        return self.source_batched.__len__()

    def __getitem__(self, index):
        batch = self.source_batched.__getitem__(index)

        # Apply transformations / augmentations
        if self.transforms is not None:
            for transform_module in self.transforms:
                batch = transform_module.forward(batch)

        return batch
