from itertools import accumulate
import torch
from torchmetrics import Metric


class L2P(Metric):
    def __init__(self, x_mean: torch.Tensor, x_std: torch.Tensor, compute_on_step=True, dist_sync_on_step=False):
        super().__init__(compute_on_step=compute_on_step, dist_sync_on_step=dist_sync_on_step)

        self.add_state("accumulated", default=torch.FloatTensor([0.0]), dist_reduce_fx="sum")
        self.add_state("count", default=torch.LongTensor([0]), dist_reduce_fx="sum")

        self.x_mean = x_mean
        self.x_std = x_std

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # We assume 3D input B x T x J*3

        target_normalized = (target - self.x_mean) / self.x_std
        preds_normalized = (preds - self.x_mean) / self.x_std

        self.accumulated += torch.sqrt(torch.sum((target_normalized - preds_normalized) ** 2.0, axis=-1)).sum()
        self.count += target.shape[0] * target.shape[1]

    def compute(self):
        return torch.mean(self.accumulated / self.count)
