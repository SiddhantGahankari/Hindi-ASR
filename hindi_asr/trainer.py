import os
import time
from collections import deque
import torch
import torch.nn as nn
import wandb
from dotenv import load_dotenv
from .optimizer import get_optimizer, get_scheduler
from .checkpoints import (
    save_checkpoint,
    load_checkpoint,
    get_checkpoint_path,
    get_best_checkpoint,
    CHECKPOINT_DIR
)
from .decoder import greedy_decode, compute_wer, compute_cer, load_vocab

load_dotenv()
os.environ["WANDB_API_KEY"] = os.getenv("WANDB_API_KEY")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VOCAB_PATH = "data/vocab.json"
D_MODEL = 384
NUM_HEADS = 6
NUM_BLOCKS = 16
VOCAB_SIZE = 300
WARMUP_STEPS = 15000
GRAD_CLIP = 1.0
LOG_EVERY = 50
EPOCHS = 30


def train_epoch(model,loader,optimizer,scheduler,ctc_loss,scaler,step,epoch,vocab_dict_rev,blank_id):
    model.train()
    total_loss = 0.0
    num_batches = 0
    running_loss = 0.0
    batches_in_interval = 0
    grad_norm = torch.tensor(0.0)
    step_times = deque(maxlen=LOG_EVERY)
    epoch_start = time.time()
    accumulation_steps = 4
    for batch_idx, batch in enumerate(loader):
        step_start = time.time()
        mels, tokens, mel_lengths, token_lengths = batch
        mels = mels.to(DEVICE, non_blocking=True)
        tokens = tokens.to(DEVICE, non_blocking=True)
        mel_lengths = mel_lengths.to(DEVICE, non_blocking=True)
        token_lengths = token_lengths.to(DEVICE, non_blocking=True)
        
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            log_probs, input_lengths = model(mels, mel_lengths)
            log_probs_ctc = log_probs.transpose(0, 1)
        loss = ctc_loss(
            log_probs_ctc,
            tokens,
            input_lengths,
            token_lengths
        )
            
        # Divide ctc_loss by accumulation_steps before executing .backward()
        loss = loss / accumulation_steps
        scaler.scale(loss).backward()
        
        # Enforce optimization step strictly once every 4 steps (or at the final trailing batch step)
        did_opt_step = False
        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(loader):
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                GRAD_CLIP
            )
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scale_after = scaler.get_scale()
            # Shift scheduler.step() inside this block so it advances only on successful optimizer steps
            if scale_before <= scale_after:
                scheduler.step()
            step += 1
            did_opt_step = True
        
        current_lr = scheduler.get_last_lr()[0]
        # Scale the running_loss counter appropriately so logged outputs reflect the true, un-divided batch loss value
        loss_item = loss.item() * accumulation_steps
        total_loss += loss_item
        running_loss += loss_item
        num_batches += 1
        batches_in_interval += 1
        
        torch.cuda.synchronize()
        step_time = time.time() - step_start
        step_times.append(step_time)
        
        # We only log on steps where we ran an optimization step and step % LOG_EVERY == 0
        if did_opt_step and step % LOG_EVERY == 0:
            avg_loss = running_loss / max(batches_in_interval, 1)
            running_loss = 0.0
            batches_in_interval = 0
            avg_step_time = sum(step_times) / len(step_times)
            elapsed = time.time() - epoch_start
            
            # Calculate CER for the current batch
            with torch.no_grad():
                hyps = greedy_decode(log_probs.detach().cpu(), input_lengths.cpu(), vocab_dict_rev, blank_id)
                refs = []
                for i in range(tokens.size(0)):
                    ref_tokens = tokens[i, :token_lengths[i]].cpu().tolist()
                    refs.append("".join([vocab_dict_rev.get(t, "<unk>") for t in ref_tokens]))
                step_cer = compute_cer(hyps, refs)

            print(
                f"epoch {epoch} | "
                f"step {step} | "
                f"loss={avg_loss:.4f} | "
                f"cer={step_cer*100:.2f}% | "
                f"lr={current_lr:.6e} | "
                f"grad_norm={grad_norm:.2f} | "
                f"step_time={avg_step_time:.3f}s | "
                f"elapsed={elapsed/60:.1f}m"
            )
            wandb.log({
                "train/loss": avg_loss,
                "train/cer": step_cer,
                "train/lr": current_lr,
                "train/grad_norm": grad_norm.item(),
                "train/step_time_sec": avg_step_time,
                "step": step,
                "epoch": epoch
            })
            
    avg_epoch_loss = total_loss / max(num_batches, 1)
    return avg_epoch_loss, step

def val_epoch(model,loader,ctc_loss,vocab_dict_rev,blank_id,epoch):
    model.eval()
    total_loss = 0.0
    num_batches = 0
    all_hyps = []
    all_refs = []
    with torch.no_grad():
        for batch in loader:
            mels, tokens, mel_lengths, token_lengths = batch
            mels = mels.to(DEVICE, non_blocking=True)
            tokens = tokens.to(DEVICE, non_blocking=True)
            mel_lengths = mel_lengths.to(DEVICE, non_blocking=True)
            token_lengths = token_lengths.to(DEVICE, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                log_probs, input_lengths = model(mels, mel_lengths)
            log_probs_ctc = log_probs.transpose(0, 1)
            loss = ctc_loss(
                log_probs_ctc,
                tokens,
                input_lengths,
                token_lengths
            )
            total_loss += loss.item()
            num_batches += 1
            hyps = greedy_decode(
                log_probs.cpu(),
                input_lengths.cpu(),
                vocab_dict_rev,
                blank_id
            )
            for i, hyp in enumerate(hyps):
                ref_tokens = tokens[i, :token_lengths[i]].cpu().tolist()
                ref = "".join([
                    vocab_dict_rev.get(t, "<unk>")
                    for t in ref_tokens
                ])
                all_hyps.append(hyp)
                all_refs.append(ref)
    val_loss = total_loss / max(num_batches, 1)
    wer = compute_wer(all_hyps, all_refs)
    cer = compute_cer(all_hyps, all_refs)
    wandb.log({
        "val/loss": val_loss,
        "val/wer": wer,
        "val/cer": cer,
        "epoch": epoch
    })
    print(
        f"epoch {epoch} | "
        f"val_loss={val_loss:.4f} | "
        f"wer={wer*100:.2f}% | "
        f"cer={cer*100:.2f}%"
    )
    return val_loss, wer, cer


def train(model, train_loader, val_loader, resume=False):
    vocab_dict, vocab_dict_rev = load_vocab(VOCAB_PATH)
    blank_id = vocab_dict["<blank>"]
    optimizer = get_optimizer(model)
    scheduler = get_scheduler(optimizer)
    ctc_loss = nn.CTCLoss(
        blank=blank_id, 
        reduction="mean",
        zero_infinity=True
    )
    scaler = torch.amp.GradScaler()
    start_epoch = 0
    step = 0
    best_wer = float("inf")
    if resume:
        best_ckpt = get_best_checkpoint(CHECKPOINT_DIR)
        if best_ckpt and os.path.exists(best_ckpt):
            ep, step, best_wer = load_checkpoint(best_ckpt, model, optimizer, scheduler, DEVICE)
            start_epoch = ep + 1
            print(f"resumed from: {best_ckpt}")
        else:
            print(f"WARNING: checkpoint not found, starting fresh")

    wandb.init(
        project="hindi-asr",
        entity="siddhantgahankari-aether",
        config={
            "d_model": D_MODEL,
            "num_heads": NUM_HEADS,
            "num_blocks": NUM_BLOCKS,
            "vocab_size": VOCAB_SIZE,
            "warmup_steps": WARMUP_STEPS,
            "grad_clip": GRAD_CLIP,
            "epochs": EPOCHS,
            "device": str(DEVICE)
        }
    )

    model.to(DEVICE)
    print(f"training on device: {DEVICE}")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters()
        if p.requires_grad
    )
    print(f"total params: {total_params:,}")
    print(f"trainable params: {trainable_params:,}")
    
    for epoch in range(start_epoch, EPOCHS):
        print(f"Epoch :{epoch}\n")
        epoch_start_time = time.time()
        train_loss, step = train_epoch(
            model, train_loader, optimizer, scheduler, 
            ctc_loss, scaler, step, epoch, vocab_dict_rev, blank_id
        )
        epoch_time = time.time() - epoch_start_time
    
        val_loss, wer, cer = val_epoch(model, val_loader, ctc_loss, vocab_dict_rev, blank_id, epoch)
        print(
            f"epoch {epoch} completed | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"wer={wer*100:.2f}% | "
            f"cer={cer*100:.2f}% | "
            f"steps={step} | "
            f"time={epoch_time/60:.2f} min"
        )

        ckpt_path = get_checkpoint_path(epoch, wer)
        save_checkpoint(model,optimizer,scheduler,epoch,step,wer,train_loss,ckpt_path)
        print(f"saved checkpoint: {ckpt_path}")
        if wer < best_wer:
            best_wer = wer
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                step,
                wer,
                train_loss,
                f"{CHECKPOINT_DIR}/best.pt"
            )
            print(f"new best WER: {best_wer*100:.2f}%")
    wandb.finish()