from torch.optim.lr_scheduler import _LRScheduler


class InverseSquareRoot(_LRScheduler):
    """Decay the LR based on the inverse square root of the update number.
    During warmup::
      lrs = torch.linspace(warmup_init_lr, lr, warmup_updates)
    After warmup::
      decay_factor = lr * sqrt(cfg.warmup_updates)
      lr = decay_factor / sqrt(update_num)
    """

    def __init__(self, optimizer, warmup_updates: int = 4000, warmup_init_lr: float = -1,
                 warmup_end_lr: float = 0.0005, epoch_scale: float = 1.0, last_epoch: int = -1, verbose: bool = False):
        
        if warmup_init_lr < 0:
            self.warmup_init_lr = 0 if warmup_updates > 0 else warmup_end_lr
        else:
            self.warmup_init_lr = warmup_init_lr
        
        self.epoch_scale = epoch_scale
        self.warmup_updates = self.epoch_scale * warmup_updates
        # linearly warmup for the first cfg.warmup_updates
        self.lr_step = (warmup_end_lr - self.warmup_init_lr) / self.warmup_updates
        # then, decay prop. to the inverse square root of the update number
        self.decay_factor = warmup_end_lr * self.warmup_updates ** 0.5

        # initial learning rate
        self.lr = warmup_init_lr        
        
        super(InverseSquareRoot, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        """Update the learning rate after each update."""
        if self.epoch_scale * self.last_epoch < self.warmup_updates:
            self.lr = self.warmup_init_lr + self.epoch_scale * self.last_epoch * self.lr_step
        else:
            self.lr = self.decay_factor * (self.epoch_scale * self.last_epoch) ** -0.5
            
        return [self.lr]