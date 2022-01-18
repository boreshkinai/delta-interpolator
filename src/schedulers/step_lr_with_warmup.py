from torch.optim.lr_scheduler import _LRScheduler


class StepLrWithWarmup(_LRScheduler):
    """Decay the LR based on the inverse square root of the update number.
    During warmup::
      lrs = torch.linspace(warmup_init_lr, lr, warmup_updates)
    After warmup::
      decay_factor = lr * sqrt(cfg.warmup_updates)
      lr = decay_factor / sqrt(update_num)
    """

    def __init__(self, optimizer, warmup_updates: int = 50, warmup_init_lr: float = -1,
                 warmup_end_lr: float = 0.0005, gamma: float = 0.5, step_size: int = 200,
                 last_epoch: int = -1, verbose: bool = False):
        
        if warmup_init_lr < 0:
            self.warmup_init_lr = 0 if warmup_updates > 0 else warmup_end_lr
        else:
            self.warmup_init_lr = warmup_init_lr
        self.warmup_end_lr = warmup_end_lr
        
        self.warmup_updates = warmup_updates
        # linearly warmup for the first cfg.warmup_updates
        self.lr_step = (self.warmup_end_lr - self.warmup_init_lr) / self.warmup_updates
        # then, apply decay rate
        self.gamma = gamma
        # initial learning rate
        self.lr = warmup_init_lr   
        self.step_size = step_size
        
        super(StepLrWithWarmup, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        """Update the learning rate after each update."""
        if self.last_epoch < self.warmup_updates:
            self.lr = self.warmup_init_lr + self.last_epoch * self.lr_step
        else:
            self.lr = self.warmup_end_lr * self.gamma ** ((self.last_epoch - self.warmup_updates) // self.step_size)
            
        return [self.lr]