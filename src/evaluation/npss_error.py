import torch
import torch.fft
from torchmetrics import Metric


class NPSS(Metric):
    def __init__(self, compute_on_step=True, dist_sync_on_step=False, eps=1e-9):
        super().__init__(compute_on_step=compute_on_step, dist_sync_on_step=dist_sync_on_step)

        self.add_state("gt_power", default=[], dist_reduce_fx="cat")
        self.add_state("pred_power", default=[], dist_reduce_fx="cat")
        self.add_state("n_frames", default=[], dist_reduce_fx=None)
        self.eps = eps

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        Computes Normalized Power Spectrum Similarity (NPSS).
        :param target: ground-truth array of shape : (Batchsize, Timesteps, Dimension)
        :param preds: shape : (Batchsize, Timesteps, Dimension)
        :return: The average NPSS metric for the batch
        """

        preds = preds.reshape(preds.shape[0], preds.shape[1], -1)
        target = target.reshape(target.shape[0], target.shape[1], -1)

        assert preds.shape == target.shape

        # Fourier coefficients along the time dimension
        gt_fourier_coeffs = torch.real(torch.fft.fft(target, dim=1))
        pred_fourier_coeffs = torch.real(torch.fft.fft(preds, dim=1))

        # Square of the Fourier coefficients
        self.n_frames = list(target.shape[1:])  # Used to reshape flattened coefficients back to original shape
        self.gt_power += torch.square(gt_fourier_coeffs)
        self.pred_power += torch.square(pred_fourier_coeffs)

    def compute(self):
        self.n_frames.insert(0, -1)
        gt_power = torch.cat(self.gt_power).reshape(*self.n_frames)
        pred_power = torch.cat(self.pred_power).reshape(*self.n_frames)

        # Sum of powers over time dimension
        gt_total_power = torch.sum(gt_power, axis=1)
        pred_total_power = torch.sum(pred_power, axis=1)

        # Normalize powers with totals
        gt_norm_power = gt_power / (gt_total_power + self.eps).unsqueeze(1)
        pred_norm_power = pred_power / (pred_total_power + self.eps).unsqueeze(1)

        # Cumulative sum over time
        cdf_gt_power = torch.cumsum(gt_norm_power, axis=1)
        cdf_pred_power = torch.cumsum(pred_norm_power, axis=1)

        # Earth mover distance
        emd = torch.linalg.norm((cdf_pred_power - cdf_gt_power), ord=1, axis=1)
        npss = torch.sum(emd * gt_total_power) / gt_total_power.sum()

        return npss
