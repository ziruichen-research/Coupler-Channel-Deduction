from torch.optim.lr_scheduler import _LRScheduler, StepLR, MultiStepLR, CosineAnnealingLR

class TransformerWarmupScheduler(_LRScheduler):
    """
    复现原项目中的 Warmup 策略：
    lr = (d_model^-0.5) * min(step^-0.5, step * warmup^-1.5)
    注意：此策略计算的是绝对 LR 值，因此 Optimizer 的初始 LR 建议设为 1.0
    """

    def __init__(self, optimizer, d_model, warmup_steps=4000, last_epoch=-1):
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        # self.last_epoch 在 step() 后会自动 +1，初始为 -1，第一次 step 后为 0
        # 我们的公式 step 从 1 开始
        step = self.last_epoch + 1
        if step <= 0: step = 1

        scale = (self.d_model ** -0.5)
        lr_val = scale * min(step ** -0.5, step * (self.warmup_steps ** -1.5))

        return [lr_val for _ in self.optimizer.param_groups]


