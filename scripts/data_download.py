from datasets import load_dataset
from dotenv import load_dotenv
from collections import Counter
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

load_dotenv()
hf_token = os.getenv("HF_KEY")

#CONFIGS --> 

SAMPLE_RATE = 16000
SAVE_DIR = "processed/mels"
MANIFEST_PATH = "manifest.jsonl"
CHECKPOINT_PATH = "pipeline_checkpoint.json"
STATS_DIR = "stats"
BATCH_SIZE = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(STATS_DIR, exist_ok=True)

vocab = set()
char_counts = Counter()
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


def replace_num_with_words(match): #self-explantory 
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
    
def encode(text, vocab_dict): #Encoder 
    return [vocab_dict.get(ch, vocab_dict["<unk>"]) for ch in text]  
    
def decode(tokens, vocab_dict_rev): #decoder 
    return "".join([vocab_dict_rev[t] for t in tokens])
    
print("making Vocab") #Start with vocab
vocab_pbar = tqdm(total=TARGET_HOURS * len(DATASETS), desc="Building Vocab", unit="hr" , dynamic_ncols=True) #TQDM setip 
for dataset_name, config in DATASETS:
    dataset = load_dataset(dataset_name, config, split="train", streaming=True, token=hf_token)
    processed_seconds = 0
    for sample in dataset:
        if processed_seconds >= TARGET_SECONDS:
            break
        text = sample["text"]
        text = normalize_text(text)
        if not text.strip():
            continue
        duration = sample["duration"]
        if duration < 1 or duration > 15:
            continue
        for ch in text:
            vocab.add(ch)
            char_counts[ch] += 1
        processed_seconds += duration
        vocab_pbar.update(duration / 3600)
        vocab_pbar.set_postfix({"vocab_size": len(vocab)})
vocab_pbar.close()
vocab.discard(" ")
vocab_s = sorted(vocab)
special_tokens = ["<blank>", "<unk>", " "]
final_vocab = special_tokens + vocab_s
vocab_dict = {token: idx for idx, token in enumerate(final_vocab)}
vocab_dict_rev = {idx: token for idx, token in enumerate(final_vocab)}
with open("vocab.json", "w+") as f:
    json.dump(vocab_dict, f, ensure_ascii=False, indent=2)
print(f"Vocabulary built: {len(vocab_dict)} tokens")

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
                "tokens": encode(batch_metadata[i]["text"], vocab_dict),
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

spectrogram_pbar.close()
manifest_file.close()

print("computing stats") #conputing the final stats of data 

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