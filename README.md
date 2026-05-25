# Hindi ASR — Conformer-CTC from Scratch

End-to-end Hindi automatic speech recognition system built from scratch using a Conformer-CTC architecture. Trained on ~600 hours of Hindi speech sourced from [AI4Bharat](https://ai4bharat.org/) datasets via HuggingFace.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [Data Sources](#3-data-sources)
4. [Data Pipeline](#4-data-pipeline)
   - [Stage 1 — Spectrogram Generation](#stage-1--spectrogram-generation)
   - [Stage 2 — BPE Tokenizer Training & Splitting](#stage-2--bpe-tokenizer-training--splitting)
   - [Stage 3 — Global Statistics](#stage-3--global-stats)
5. [Processed Data Layout](#5-processed-data-layout)
6. [Model Architecture](#6-model-architecture)
7. [Training Pipeline](#7-training-pipeline)
   - [DataLoader & Bucketing](#dataloader--bucketing)
   - [Learning Rate Schedule](#learning-rate-schedule)
   - [Loss Function](#loss-function)
   - [Mixed Precision](#mixed-precision)
   - [Checkpointing](#checkpointing)
8. [Hyperparameter Reference](#8-hyperparameter-reference)
9. [Environment Setup](#9-environment-setup)
10. [Running the Project](#10-running-the-project)
11. [Monitoring](#11-monitoring)
12. [Inference](#12-inference)

---

## 1. Project Overview

| Property | Value |
|---|---|
| Task | Hindi Automatic Speech Recognition (ASR) |
| Architecture | Conformer-CTC |
| Parameters | ~61 million |
| Target data | ~600 hours Hindi speech |
| Loss | CTC (Connectionist Temporal Classification) |
| Decoder | Greedy (greedy search) |
| Vocabulary | 300 tokens (BPE subwords + specials) |
| Framework | PyTorch |

---

## 2. Repository Structure

```
Hindi-ASR/
│
├── main.py               # Root entrypoint for data/train/infer
├── hindi_asr/
│   ├── __init__.py
│   ├── checkpoints.py      # save/load checkpoint helpers
│   ├── dataset.py          # PyTorch Dataset, BucketSampler, collate_fn
│   ├── decoder.py          # Greedy decoder, WER, CER computation
│   ├── model.py            # Conformer-CTC architecture
│   ├── optimizer.py        # AdamW + Noam LR scheduler
│   └── trainer.py          # train_epoch / val_epoch / train loop
│
├── scripts/
│   ├── data_download.py    # Full data pipeline: download → preprocess → save
│   ├── train.py            # Entry point — builds model & calls train()
│   ├── inference.py        # Single-file inference script
│   └── main.py             # Alternate entry (thin wrapper)
│
├── data/
│   ├── vocab.json              # BPE subword vocabulary mapping (300 tokens)
│   ├── manifest_train.jsonl    # One JSON line per training sample
│   ├── manifest_val.jsonl      # One JSON line per validation sample
│   └── stats/
│       └── global_stats.npz    # Per-mel-bin mean, variance, std
│
├── processed/
│   └── mels/
│       ├── 000000.npy          # Log-mel spectrogram, shape [80, T], float16
│       ├── 000001.npy
│       └── ...
│
├── checkpoints/
│   ├── best.pt                 # Best checkpoint by val WER
│   ├── epoch000_wer0.9813.pt
│   └── ...
│
├── pipeline_checkpoint.json    # Resume state for data_download.py
├── train.log                   # Training stdout log
└── .env                        # HF_KEY, WANDB_API_KEY
```

---

## 3. Data Sources

Data is streamed directly from HuggingFace Hub using the `datasets` library in **streaming mode** (no full download to disk). Two AI4Bharat datasets are used:

| Dataset | HuggingFace ID | Config | Split |
|---|---|---|---|
| Shrutilipi | `ai4bharat/Shrutilipi` | `hindi` | `train` |
| IndicVoices | `ai4bharat/indicvoices` | `hindi` | `train` |

**Authentication**: Both datasets require a HuggingFace API token stored in `.env` as `HF_KEY`.

**Target volume**: 300 hours per dataset (`TARGET_HOURS = 300`, `processed_seconds` resets for each dataset) → **~600 hours total** after quality filtering (SNR, duration, empty transcript, NaN).

Each raw sample from HuggingFace contains:
```
{
  "audio_filepath": <audio bytes object>,
  "text":           <raw Hindi transcript string>,
  "duration":       <float, seconds>
}
```

---

## 4. Data Pipeline

All processing is done in `scripts/data_download.py`. It runs in three sequential stages. The pipeline is **resumable** — if interrupted, it reads `pipeline_checkpoint.json` and skips already-processed samples.

### Stage 1 — Spectrogram Generation

**Purpose**: Convert raw audio waveforms into log-mel spectrograms and save them as compressed numpy arrays alongside a temporary manifest.

**Batch processing**: Waveforms are accumulated into batches of 128 (`BATCH_SIZE=128`) and processed together on GPU for efficiency. Zero-padding is applied via `pad_sequence` to make waveforms the same length within each batch.

**Audio processing per sample**:
1. **Load audio** — decode bytes from the HuggingFace audio object
2. **Mono conversion** — average channels if stereo: `waveform.mean(dim=0, keepdim=True)`
3. **SNR filter** — reject samples with Signal-to-Noise Ratio < 20 dB:
   - Estimate noise floor from the lowest-energy 10% of frames
   - Estimate signal level from the highest-energy 10% of frames
   - `SNR = 10 × log10(signal / noise)` — reject if < 20 dB
4. **Resample** — convert to 16kHz if original sample rate differs (resampler cached per source rate)
5. **Mel spectrogram** (on GPU):

   | Parameter | Value |
   |---|---|
   | Sample rate | 16,000 Hz |
   | FFT size (n_fft) | 400 (25ms window) |
   | Window length | 400 samples |
   | Hop length | 160 samples (10ms stride) |
   | Mel bins | 80 |
   | f_min | 80 Hz |
   | f_max | 7,600 Hz |

6. **Log compression** — `AmplitudeToDB()` with a small epsilon (`+1e-9`) for numerical stability
7. **NaN check** — reject if any spectrogram value is NaN
8. **Save** — stored as `float16` numpy array to `processed/mels/{index:06d}.npy`, shape: `[80, T]`

**Temporary Manifest entry** written to `manifest.jsonl` per saved sample:
```json
{
  "mel_path": "processed/mels/000042.npy",
  "text":     "यह एक परीक्षण वाक्य है",
  "duration": 3.72,
  "dataset":  "ai4bharat/Shrutilipi"
}
```

**Rejection counters tracked**:
| Reason | Counter |
|---|---|
| Duration out of [1s, 15s] | `rejected_duration` |
| SNR < 20 dB | `rejected_snr` |
| Empty transcript after normalization | `rejected_empty` |
| NaN in spectrogram | `rejected_nan` |

**Checkpoint** saved every 500 samples to `pipeline_checkpoint.json`:
```json
{ "saved_samples": 45000, "processed_seconds_total": 486321.4 }
```

---

### Stage 2 — BPE Tokenizer Training & Splitting

**Purpose**: Train a Byte-Pair Encoding (BPE) subword tokenizer on the normalized transcripts, encode them into subword tokens, and split the dataset automatically.

**Flow**:
1. Read all transcripts from temporary `manifest.jsonl`.
2. Train a HuggingFace `tokenizers` BPE model of size 300 with special tokens:
   - `<blank>` (index 0) — CTC blank token
   - `<unk>` (index 1) — unknown subword fallback
   - `" "` (index 2) — space character
3. Save BPE vocabulary mapping of size 300 to `data/vocab.json`.
4. Encode the transcripts into token sequences using the trained BPE model.
5. Shuffle the dataset with seed 42, and split into:
   - **Train set (90%)**: `data/manifest_train.jsonl`
   - **Val set (10%)**: `data/manifest_val.jsonl`
6. Clean up the temporary `manifest.jsonl`.

---

### Stage 3 — Global Statistics

**Purpose**: Compute per-mel-bin mean and variance across the entire dataset for use in training-time normalization.

Iterates over all saved `.npy` files and accumulates:
```
sum_acc[bin]    += mel[bin, :].sum()
sq_sum_acc[bin] += (mel[bin, :] ** 2).sum()
total_frames    += T
```

Final statistics:
```
global_mean[bin] = sum_acc[bin] / total_frames
global_var[bin]  = (sq_sum_acc[bin] / total_frames) - global_mean[bin]²
```

**Output** — `data/stats/global_stats.npz`:
```
mean  → shape [80]   (per-bin mean)
var   → shape [80]   (per-bin variance)
std   → shape [80]   (per-bin std = √var)
```

This file is loaded by `HindiASRDataset` at training time to normalize each mel spectrogram:
```python
mel = (mel - mean[:, None]) / std[:, None]
```

---

## 5. Processed Data Layout

```
data/
├── vocab.json                  # 300-token BPE subword vocabulary
├── manifest_train.jsonl        # ~90% of samples, one JSON per line
├── manifest_val.jsonl          # ~10% of samples
└── stats/
    └── global_stats.npz        # mean[80], var[80], std[80]

processed/
└── mels/
    ├── 000000.npy              # float16, shape [80, T]
    ├── 000001.npy
    └── ...                     # ~300k+ files
```

Each `.npy` file stores a single utterance as a log-mel spectrogram:
- **dtype**: `float16` (saves ~50% disk vs float32)
- **shape**: `[80, T]` where T = `ceil(audio_samples / hop_length)` after subsampling
- **loaded as**: `float32` at training time (`astype(np.float32)`)

---

## 6. Model Architecture

`model.py` implements a standard **Conformer-CTC** encoder with a linear CTC head. No language model or decoder is attached — pure CTC greedy decoding at inference.

### Conv2D Subsampler (`Conv2DSubsampler`)

Reduces the time dimension by **4×** before the Conformer blocks:

```
Input:  [B, 80, T]
→ unsqueeze + transpose → [B, 1, T, 80]
→ Conv2d(1, 384, 3, stride=2) + ReLU   → [B, 384, T/2, 40]
→ Conv2d(384, 384, 3, stride=2) + ReLU → [B, 384, T/4, 20]
→ reshape + Linear(384×20 → 384)       → [B, T/4, 384]
```

CTC input lengths are computed as `mel_lengths // 4`.

### Relative Positional Encoding (`RelativePositionalEncoding`)

Sinusoidal positional encoding with `max_len=5000` frames (post-subsampling), stored as a non-trainable buffer. Used in the Transformer-XL style relative attention computation.

### Conformer Block (`ConformerBlock`)

Each block follows the standard Conformer structure:

```
x → FF (½ scale) → Self-Attention (RelPE) → Convolution → FF (½ scale) → LayerNorm
```

Components per block:
- **FeedForward** (`FeedForwardModule`): Pre-norm LayerNorm → Linear(384→1536) → SiLU → Dropout → Linear(1536→384) → Dropout → residual × 0.5
- **Multi-Head Self-Attention** (`MultiHeadSelfAttentionWithRelPE`): Pre-norm LayerNorm → 6-head relative position attention (Transformer-XL style with learnable `u_bias`, `v_bias`) → Dropout → residual
- **Convolution** (`ConvolutionModule`): Pre-norm LayerNorm → Pointwise Conv (GLU) → Depthwise Conv(kernel=31) → BatchNorm → SiLU → Pointwise Conv → Dropout → residual

### CTC Head

```
Linear(384 → 300) → log_softmax(dim=-1)
```

### Full Model Summary

| Component | Config |
|---|---|
| d_model | 384 |
| num_heads | 6 |
| head_dim | 64 |
| num_blocks | 16 |
| FFN expansion | 4× (384 → 1536) |
| Conv kernel size | 31 |
| Dropout | 0.15 |
| **Total parameters** | **~61.3 million** |

---

## 7. Training Pipeline

### DataLoader & Bucketing

`BucketSampler` bins samples by duration into 5 buckets defined by `BUCKET_BOUNDARIES = [3, 6, 9, 12]` seconds:

| Bucket | Duration range |
|---|---|
| 0 | < 3s |
| 1 | 3–6s |
| 2 | 6–9s |
| 3 | 9–12s |
| 4 | > 12s |

Within each bucket, samples are shuffled and batched together. The batch order across buckets is also shuffled each epoch. This minimises padding waste compared to random sampling.

`collate_fn` zero-pads mel spectrograms to `max_T` and token sequences to `max_L` within each batch.

**DataLoader config**:
- `batch_size = 16` (set in `train.py`)
- `num_workers = 4`
- `pin_memory = True`

### Learning Rate Schedule

**Noam (inverse-square-root) warmup schedule** as defined in the original Transformer paper:

```
lr(step) = PEAK_LR × min(step / warmup, √(warmup / step))
```

| Step | LR |
|---|---|
| 0 | 0 |
| 2000 | 5.0e-5 |
| **4000** | **1.0e-4 (peak)** |
| 9155 (end epoch 0) | ~6.6e-5 |
| 18310 (end epoch 1) | ~4.7e-5 |
| 27465 (end epoch 2) | ~3.8e-5 |

### Loss Function

`nn.CTCLoss` with:
- `blank=0` (`<blank>` token at index 0)
- `reduction="mean"` — loss averaged over batch
- `zero_infinity=True` — silently ignores infinite losses (guards against very short inputs)

### Mixed Precision

`torch.amp.GradScaler` + `torch.autocast(device_type="cuda", dtype=torch.bfloat16)` applied to the forward pass. Gradients are unscaled before clipping.

```python
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    log_probs, input_lengths = model(mels, mel_lengths)
    loss = ctc_loss(...)

scaler.scale(loss).backward()
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)  # clip at 1.0
scaler.step(optimizer)
scaler.update()
```

### Data Augmentation (SpecAugment)

Applied **online** during training only (not validation) in `HindiASRDataset.__getitem__`:

| Mask type | Count | Parameter |
|---|---|---|
| Frequency masking | 2 | `freq_mask_param=27` (~34% of 80 bins max) |
| Time masking | 2 | `time_mask_param=min(100, T//2)` |

### Checkpointing

Every epoch saves two files:
- `checkpoints/epoch{N:03d}_wer{wer:.4f}.pt` — per-epoch checkpoint
- `checkpoints/best.pt` — overwritten when val WER improves

Each checkpoint contains:
```python
{
    "epoch":            int,
    "step":             int,
    "best_wer":         float,
    "loss":             float,
    "model_state":      OrderedDict,
    "optimizer_state":  dict,
    "scheduler_state":  dict
}
```

---

## 8. Hyperparameter Reference

### Optimizer (`hindi_asr/optimizer.py`)

| Parameter | Value | Notes |
|---|---|---|
| Optimizer | AdamW | |
| `PEAK_LR` | `1e-4` | Noam peak learning rate |
| `WARMUP_STEPS` | `4000` | ~0.44 epochs |
| `betas` | `(0.9, 0.98)` | Lower β₂ for better CTC stability |
| `eps` | `1e-9` | |
| `weight_decay` (weights) | `5e-3` | Applied to all non-norm/non-bias params |
| `weight_decay` (norms/biases) | `0.0` | Excluded from decay |

### Trainer (`hindi_asr/trainer.py`)

| Parameter | Value | Notes |
|---|---|---|
| `GRAD_CLIP` | `1.0` | Standard for Conformer/CTC |
| `EPOCHS` | `30` | |
| `LOG_EVERY` | `50` | Steps between log prints |
| `D_MODEL` | `384` | |
| `NUM_HEADS` | `6` | |
| `NUM_BLOCKS` | `16` | |
| `VOCAB_SIZE` | `300` | |

### Dataset (`hindi_asr/dataset.py` / `scripts/train.py`)

| Parameter | Value |
|---|---|
| `BATCH_SIZE` | `32` |
| `BUCKET_BOUNDARIES` | `[3, 6, 9, 12]` seconds |
| Freq mask param | `27` |
| Time mask param | `min(100, T//2)` |
| Freq mask count | `2` |
| Time mask count | `2` |

---

## 9. Environment Setup

### Requirements

```bash
pip install torch torchaudio datasets transformers
pip install python-dotenv wandb
pip install indic-nlp-library indic-numtowords
pip install tqdm numpy
```

Alternatively, install the package for editable development:

```bash
pip install -e .
```

### `.env` file

Create a `.env` file in the project root:

```env
HF_KEY=hf_your_huggingface_token_here
WANDB_API_KEY=your_wandb_api_key_here
```

- `HF_KEY` — HuggingFace token with access to `ai4bharat/Shrutilipi` and `ai4bharat/indicvoices`
- `WANDB_API_KEY` — Weights & Biases key for run tracking

---

## 10. Running the Project

### Step 1 — Data Pipeline

Run once. This downloads, filters, and preprocesses the full dataset. **Takes several hours** depending on internet speed and GPU.

```bash
python scripts/data_download.py
```

Or via the root entrypoint:

```bash
python main.py data
```

The script is **fully resumable**. If interrupted, re-run the same command — it reads `pipeline_checkpoint.json` and skips already-processed samples.

Expected output at completion:
```
Saved:              ~300000 spectrograms
Processed:          ~600.00 hours
Vocabulary size:    300 tokens
Rejected duration:  XXXXX
Rejected SNR:       XXXXX
Rejected empty:     XXXXX
Rejected NaN:       XXXXX
Stats saved to:     data/stats/global_stats.npz
```

No manual steps are required! The pipeline automatically shuffles and splits the dataset, trains the BPE tokenizer, and stores all artifacts directly in the `data/` directory.

### Step 2 — Training (from scratch)

Ensure `scripts/train.py` has `resume=False`:

```bash
python scripts/train.py
```

Or via the root entrypoint:

```bash
python main.py train
```

To resume from the best saved checkpoint, set `resume=True` in `train.py`. The resume block loads only model weights — optimizer and scheduler reset fresh.

### Step 3 — Inference

```bash
python scripts/inference.py --audio path/to/audio.wav
```

Or via the root entrypoint:

```bash
python main.py infer --audio path/to/audio.wav
```

---

## 11. Monitoring

Training is tracked via [Weights & Biases](https://wandb.ai) under project `hindi-asr`.

Metrics logged every `LOG_EVERY=50` steps:

| Metric | Description |
|---|---|
| `train/loss` | Average CTC loss over last 50 steps |
| `train/cer` | Character Error Rate on current batch |
| `train/lr` | Current learning rate |
| `train/grad_norm` | Gradient norm before clipping |
| `train/step_time_sec` | Average seconds per step |
| `val/loss` | Validation CTC loss (per epoch) |
| `val/wer` | Validation Word Error Rate (per epoch) |
| `val/cer` | Validation Character Error Rate (per epoch) |

WandB run: https://wandb.ai/siddhantgahankari-aether/hindi-asr

### Expected Training Behavior (healthy run)

| Epoch | Expected loss | Expected CER |
|---|---|---|
| 0 | 27 → 3–4 | 100% → 60–80% |
| 1 | ~3.0–3.3 | ~50–70% |
| 2–4 | dropping | alignment "click" — CER drops sharply |
| 5+ | steady descent | WER begins to be meaningful |

---

## 12. Inference

`inference.py` loads the best checkpoint and runs greedy CTC decoding on a WAV file.

```python
from inference import transcribe
text = transcribe("path/to/audio.wav")
print(text)
```

**Preprocessing at inference**: same mel spectrogram transform (400/160/80) → same global stats normalization → model forward → greedy CTC decode (collapse repeats, remove blanks).

No beam search or language model is currently used. WER can be further improved by adding a KenLM language model with beam search (e.g., `pyctcdecode`).

---

## Notes

- The `dataset.py` file defines `HindiASRDataset`, `BucketSampler`, and `collate_fn` as importable classes only. It does **not** instantiate any DataLoader at module level — all DataLoaders are created in `train.py`.
- Spectrograms are saved as `float16` to save disk space and loaded as `float32` for training.
- The `<blank>` token is always at index 0, which is the required blank index for `nn.CTCLoss`.
- `zero_infinity=True` in CTCLoss silently handles edge cases where the input sequence is too short for the target — important for very short audio clips that slip through the duration filter.
