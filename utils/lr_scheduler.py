from torch.optim.lr_scheduler import _LRScheduler, StepLR, MultiStepLR, CosineAnnealingLR


class TransformerWarmupScheduler(_LRScheduler):
    """
    Transformer-style warmup schedule:
        lr = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))

    This returns absolute LR values, so using optimizer base lr=1.0 is typical.
    """

    def __init__(self, optimizer, d_model, warmup_steps=4000, last_epoch=-1):
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        # _LRScheduler starts from last_epoch=-1; use step index starting at 1.
        step = self.last_epoch + 1
        if step <= 0:
            step = 1

        scale = self.d_model ** -0.5
        lr_val = scale * min(step ** -0.5, step * (self.warmup_steps ** -1.5))
        return [lr_val for _ in self.optimizer.param_groups]
