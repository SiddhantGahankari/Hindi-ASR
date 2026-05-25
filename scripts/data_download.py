from datasets import load_dataset
from dotenv import load_dotenv
import os
import re
import string
from indic_numtowords import num2words
import torchaudio
import numpy as np
import torch
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
import json
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from pathlib import Path
import random
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

load_dotenv()
hf_token = os.getenv("HF_KEY")

#CONFIGS --> 

SAMPLE_RATE = 16000
SAVE_DIR = "processed/mels"
MANIFEST_PATH = "manifest.jsonl"
CHECKPOINT_PATH = "pipeline_checkpoint.json"
STATS_DIR = "data/stats"
BATCH_SIZE = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(STATS_DIR, exist_ok=True)

TARGET_HOURS = 300
TARGET_SECONDS = TARGET_HOURS * 3600
saved_samples = 0
processed_seconds_total = 0
rejected_duration = 0
rejected_snr = 0
rejected_empty = 0
rejected_nan = 0

DATASETS = [
    ("ai4bharat/Shrutilipi", "hindi"),
    ("ai4bharat/indicvoices", "hindi"),
]

#TRANSFORMS -->

mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000,
    n_fft=400,
    win_length=400,
    hop_length=160,
    n_mels=80,
    f_min=80,
    f_max=7600
).to(DEVICE)

db_transform = torchaudio.transforms.AmplitudeToDB().to(DEVICE)


#NORMALIZATION

_factory = IndicNormalizerFactory()
_normalizer = _factory.get_normalizer("hi", remove_nuktas=False)


def replace_num_with_words(match): #self-explanatory 
    num_val = int(match.group())
    return num2words(num_val, lang='hi')


def normalize_text(text): #basic text normalization 
    text = _normalizer.normalize(text)
    text = re.sub(r'[a-zA-Z]', '', text)
    text = re.sub(r'[\(\)\[\]\{\}]', ' ', text)
    devanagari_nums = ('०','१','२','३','४','५','६','७','८','९')
    for i, num in enumerate(devanagari_nums):
        text = text.replace(num, str(i))
    text = re.sub(r'\d+', replace_num_with_words, text)
    punctuations = set(string.punctuation + '।॥')
    for p in punctuations:
        text = text.replace(p, ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    return text

print("generating spectrogram") #start with generation of spectrogram

#RESUME CHECK if the server dies in middle of this T_T(very probable)
if os.path.exists(CHECKPOINT_PATH):
    with open(CHECKPOINT_PATH) as f:
        ckpt = json.load(f)
    saved_samples = ckpt["saved_samples"]
    processed_seconds_total = ckpt["processed_seconds_total"]
    manifest_file = open(MANIFEST_PATH, "a", encoding="utf-8")
    print(f"Resuming from checkpoint: {saved_samples} samples, {processed_seconds_total/3600:.2f} hours")
else:
    manifest_file = open(MANIFEST_PATH, "w", encoding="utf-8")


resamplers = {}
spectrogram_pbar = tqdm(
    total=TARGET_HOURS * len(DATASETS),
    desc="Saving Spectrograms",
    unit="hr",
    dynamic_ncols=True
)


samples_to_skip = saved_samples
skipped = 0

batch_waveforms = []
batch_metadata = []

for dataset_name, config in DATASETS:
    dataset = load_dataset(dataset_name, config, split="train", streaming=True, token=hf_token)
    processed_seconds = 0
    for sample in dataset:
        if skipped < samples_to_skip:
            skipped += 1
            continue
        if processed_seconds >= TARGET_SECONDS:
            break
        text = sample["text"]
        text = normalize_text(text)
        if not text.strip():
            rejected_empty += 1
            continue
            
        duration = sample["duration"]
        if duration < 1 or duration > 15:
            rejected_duration += 1
            continue
            
        path = sample["audio_filepath"]
        decoded = path.get_all_samples()
        waveform = torch.tensor(decoded.data).float()
        sample_rate = decoded.sample_rate
        
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.shape[0] != 1:
            waveform = waveform.mean(dim=0, keepdim=True)
            
        #SNR calculation 
        frame_energies = waveform.pow(2).mean(dim=0)
        sorted_energies = frame_energies.sort().values
        num_noise_frames = max(1, len(sorted_energies) // 10)
        noise_floor = sorted_energies[:num_noise_frames].mean() + 1e-9
        signal_level = sorted_energies[-num_noise_frames:].mean()
        snr = 10 * torch.log10(signal_level / noise_floor)
        if snr < 20:
            rejected_snr += 1
            continue
            
        #Sample rate --> 16KHz set
        if sample_rate != SAMPLE_RATE:
            if sample_rate not in resamplers:
                resamplers[sample_rate] = torchaudio.transforms.Resample(
                    orig_freq=sample_rate,
                    new_freq=SAMPLE_RATE
                )
            waveform = resamplers[sample_rate](waveform)

        batch_waveforms.append(waveform.squeeze(0))

        batch_metadata.append({
            "text": text,
            "duration": duration,
            "dataset": dataset_name
        })

        if len(batch_waveforms) < BATCH_SIZE:
            continue

        padded_waveforms = pad_sequence(
            batch_waveforms,
            batch_first=True
        ).to(DEVICE)

        mel_spec = mel_transform(padded_waveforms)

        log_mel = db_transform(mel_spec + 1e-9)

        log_mels = log_mel.cpu()

        for i in range(len(batch_metadata)):

            current_log_mel = log_mels[i]

            if torch.isnan(current_log_mel).any():
                rejected_nan += 1
                continue

            save_path = os.path.join(SAVE_DIR, f"{saved_samples:06d}.npy") #saving them 

            np.save(
                save_path,
                current_log_mel.numpy().astype(np.float16)
            )

            manifest_entry = {
                "mel_path": save_path,
                "text": batch_metadata[i]["text"],
                "duration": batch_metadata[i]["duration"],
                "dataset" : batch_metadata[i]["dataset"]
            }

            manifest_file.write(json.dumps(manifest_entry, ensure_ascii=False) + "\n")

            processed_seconds += batch_metadata[i]["duration"]
            processed_seconds_total += batch_metadata[i]["duration"]
            saved_samples += 1

            spectrogram_pbar.update(batch_metadata[i]["duration"] / 3600)

            spectrogram_pbar.set_postfix({
                "saved": saved_samples,
                "hours": round(processed_seconds_total / 3600, 2),
            })

            if saved_samples % 500 == 0:
                manifest_file.flush()
                with open(CHECKPOINT_PATH, "w") as f:
                    json.dump({
                        "saved_samples": saved_samples,
                        "processed_seconds_total": processed_seconds_total
                    }, f)

        batch_waveforms = []
        batch_metadata = []

# Process remaining items in buffer if any
if len(batch_waveforms) > 0:
    padded_waveforms = pad_sequence(
        batch_waveforms,
        batch_first=True
    ).to(DEVICE)
    mel_spec = mel_transform(padded_waveforms)
    log_mel = db_transform(mel_spec + 1e-9)
    log_mels = log_mel.cpu()
    for i in range(len(batch_metadata)):
        current_log_mel = log_mels[i]
        if torch.isnan(current_log_mel).any():
            rejected_nan += 1
            continue
        save_path = os.path.join(SAVE_DIR, f"{saved_samples:06d}.npy")
        np.save(
            save_path,
            current_log_mel.numpy().astype(np.float16)
        )
        manifest_entry = {
            "mel_path": save_path,
            "text": batch_metadata[i]["text"],
            "duration": batch_metadata[i]["duration"],
            "dataset" : batch_metadata[i]["dataset"]
        }
        manifest_file.write(json.dumps(manifest_entry, ensure_ascii=False) + "\n")
        processed_seconds_total += batch_metadata[i]["duration"]
        saved_samples += 1
        spectrogram_pbar.update(batch_metadata[i]["duration"] / 3600)
        spectrogram_pbar.set_postfix({
            "saved": saved_samples,
            "hours": round(processed_seconds_total / 3600, 2),
        })

spectrogram_pbar.close()
manifest_file.close()

# ----------------- Stage 2: Tokenizer Training & Splitting -----------------
print("Training BPE tokenizer...")
transcripts = []
if not os.path.exists(MANIFEST_PATH):
    raise FileNotFoundError(f"Master manifest file not found at: {MANIFEST_PATH}")
    
with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            entry = json.loads(line)
            transcripts.append(entry["text"])
            
print(f"Loaded {len(transcripts)} transcripts for training.")

tokenizer = Tokenizer(BPE(unk_token="<unk>"))
tokenizer.pre_tokenizer = Whitespace()

special_tokens = ["<blank>", "<unk>", " "]
trainer = BpeTrainer(
    vocab_size=300,
    special_tokens=special_tokens
)

tokenizer.train_from_iterator(transcripts, trainer=trainer)

vocab = tokenizer.get_vocab()
assert vocab.get("<blank>") == 0, f"Expected <blank> at index 0, got {vocab.get('<blank>')}"
assert vocab.get("<unk>") == 1, f"Expected <unk> at index 1, got {vocab.get('<unk>')}"
assert vocab.get(" ") == 2, f"Expected ' ' at index 2, got {vocab.get(' ')}"
print("Special token indices successfully verified!")

os.makedirs("data", exist_ok=True)
sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
vocab_dict = {token: idx for token, idx in sorted_vocab}

vocab_path = "data/vocab.json"
with open(vocab_path, "w", encoding="utf-8") as f:
    json.dump(vocab_dict, f, ensure_ascii=False, indent=2)
print(f"Saved vocabulary mapping of size {len(vocab_dict)} to {vocab_path}")

print("Tokenizing transcripts and preparing dataset splits...")
entries = []
with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            entries.append(json.loads(line))

for entry in entries:
    text = entry["text"]
    words = text.strip().split()
    tokens = []
    for i, word in enumerate(words):
        if i > 0:
            tokens.append(2)  # Insert space token index (2) between words
        tokens.extend(tokenizer.encode(word).ids)
    entry["tokens"] = tokens

random.seed(42)
random.shuffle(entries)

num_entries = len(entries)
split_idx = int(0.9 * num_entries)
train_entries = entries[:split_idx]
val_entries = entries[split_idx:]

train_path = Path("data/manifest_train.jsonl")
val_path = Path("data/manifest_val.jsonl")

with open(train_path, "w", encoding="utf-8") as f:
    for entry in train_entries:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

with open(val_path, "w", encoding="utf-8") as f:
    for entry in val_entries:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"Dataset split complete:")
print(f" - Train entries: {len(train_entries)} written to {train_path}")
print(f" - Val entries: {len(val_entries)} written to {val_path}")

# Clean up raw manifest.jsonl
if os.path.exists(MANIFEST_PATH):
    os.remove(MANIFEST_PATH)
    print(f"Cleaned up temporary master manifest: {MANIFEST_PATH}")

# ----------------- Stage 3: Global Statistics -----------------
print("computing stats") #computing the final stats of data 

sum_acc = np.zeros(80, dtype=np.float64)
sq_sum_acc = np.zeros(80, dtype=np.float64)
total_frames = 0

stats_pbar = tqdm(total=saved_samples, desc="Computing Stats", unit="file" , dynamic_ncols=True)

for i in range(saved_samples):
    mel = np.load(os.path.join(SAVE_DIR, f"{i:06d}.npy")).astype(np.float32)
    sum_acc += mel.sum(axis=1)
    sq_sum_acc += (mel ** 2).sum(axis=1)
    total_frames += mel.shape[1]
    stats_pbar.update(1)
stats_pbar.close()
global_mean = sum_acc / total_frames
global_var = (sq_sum_acc / total_frames) - (global_mean ** 2)
global_std = np.sqrt(global_var)

np.savez(
    os.path.join(STATS_DIR, "global_stats.npz"),
    mean=global_mean,
    var=global_var,
    std=global_std
)

print(f"Saved:              {saved_samples} spectrograms")
print(f"Processed:          {processed_seconds_total / 3600:.2f} hours")
print(f"Vocabulary size:    {len(vocab_dict)} tokens")
print(f"Rejected duration:  {rejected_duration}")
print(f"Rejected SNR:       {rejected_snr}")
print(f"Rejected empty:     {rejected_empty}")
print(f"Rejected NaN:       {rejected_nan}")
print(f"Stats saved to:     {STATS_DIR}/global_stats.npz")