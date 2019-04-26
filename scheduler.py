import torch
from torch.optim.lr_scheduler import _LRScheduler


class PolyLR(_LRScheduler):
    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 min_lr: float,
                 max_epoch: int,
                 power: float = 2.0,
                 last_epoch: int = -1):
        self.min_lr = min_lr
        self.max_epoch = max_epoch
        self.power = power
        super(PolyLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        coef = (1 - self.last_epoch / self.max_epoch) ** self.power
        return [self.min_lr + (base_lr - self.min_lr) * coef for base_lr in self.base_lrs]
