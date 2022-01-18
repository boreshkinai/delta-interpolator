import torch
from torchmetrics import Metric


class L2Q(Metric):
    def __init__(self, compute_on_step=True, dist_sync_on_step=False):
        super().__init__(compute_on_step=compute_on_step, dist_sync_on_step=dist_sync_on_step)

        self.add_state("accumulated", default=torch.FloatTensor([0.0]), dist_reduce_fx="sum")
        self.add_state("count", default=torch.LongTensor([0]), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        preds = preds.reshape(preds.shape[0], preds.shape[1], -1)
        target = target.reshape(target.shape[0], target.shape[1], -1)

        assert preds.shape == target.shape

        self.accumulated += torch.sqrt(torch.sum((target - preds) ** 2.0, axis=-1)).sum()
        self.count += target.shape[0] * target.shape[1]


    def compute(self):
        return torch.mean(self.accumulated / self.count)
