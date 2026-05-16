import torch
import os

CHECKPOINT_DIR = "checkpoints"
D_MODEL = 384
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def save_checkpoint(model, optimizer, scheduler, epoch, step, best_wer, loss, path):
    torch.save({
        "epoch": epoch,
        "step": step,
        "best_wer": best_wer,
        "loss": loss,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict()
    }, path)
    print(f"saved checkpoint: {path}")

def _move_optimizer_state_to_device(optimizer, device):
    """Move optimizer state tensors (e.g. AdamW exp_avg buffers) to the target device."""
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

def load_checkpoint(path, model, optimizer=None, scheduler=None, device=None):
    map_loc = device if device is not None else "cpu"
    ckpt = torch.load(path, map_location=map_loc)
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer_state"])
        if device is not None:
            _move_optimizer_state_to_device(optimizer, device)
    if scheduler is not None:
        scheduler.load_state_dict(ckpt["scheduler_state"])
    print(f"loaded checkpoint: epoch={ckpt['epoch']} step={ckpt['step']} best_wer={ckpt['best_wer']:.4f}")
    return ckpt["epoch"], ckpt["step"], ckpt["best_wer"]

def get_checkpoint_path(epoch, wer):
    return os.path.join(CHECKPOINT_DIR, f"epoch{epoch:03d}_wer{wer:.4f}.pt")

def get_best_checkpoint(checkpoint_dir):
    if not os.path.isdir(checkpoint_dir):
        return None
    best_path = os.path.join(checkpoint_dir, "best.pt")
    if os.path.exists(best_path):
        return best_path
    files = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pt") and "wer" in f]
    if not files:
        return None
    files.sort(key=lambda f: float(f.split("wer")[1].replace(".pt", "")))
    return os.path.join(checkpoint_dir, files[0])
