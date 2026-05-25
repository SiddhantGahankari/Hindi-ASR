import torch
WARMUP_STEPS = 15000
PEAK_LR = 5e-4

def get_param_groups(model):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if (
            "batch_norm" in name
            or "bias" in name
            or "norm" in name
        ):
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {
            "params": decay,
            "weight_decay": 1e-4
        },
        {
            "params": no_decay,
            "weight_decay": 0.0
        }
    ]
def get_optimizer(model):
    param_groups = get_param_groups(model)
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=PEAK_LR,
        betas=(0.9, 0.98),
        eps=1e-9
    )
    return optimizer
def lr_lambda(step):
    step = max(step, 1)
    if step <= WARMUP_STEPS:
        return step / WARMUP_STEPS
    return (WARMUP_STEPS ** 0.5) / (step ** 0.5)
def get_scheduler(optimizer):
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lr_lambda
    )
    return scheduler