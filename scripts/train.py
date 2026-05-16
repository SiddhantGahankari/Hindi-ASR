from hindi_asr.model import ConformerCTC
from hindi_asr.dataset import HindiASRDataset, BucketSampler, collate_fn
from hindi_asr.trainer import train
from torch.utils.data import DataLoader

MANIFEST_TRAIN = "data/manifest_train.jsonl"
MANIFEST_VAL = "data/manifest_val.jsonl"
STATS_PATH = "data/stats/global_stats.npz"
BATCH_SIZE = 32
BUCKET_BOUNDARIES = [3, 6, 9, 12]
D_MODEL = 384
NUM_HEADS = 6
NUM_BLOCKS = 16
VOCAB_SIZE = 71

train_dataset = HindiASRDataset(MANIFEST_TRAIN, STATS_PATH, is_train=True)
val_dataset = HindiASRDataset(MANIFEST_VAL, STATS_PATH, is_train=False)

train_sampler = BucketSampler(train_dataset, BATCH_SIZE, BUCKET_BOUNDARIES, shuffle=True)
val_sampler = BucketSampler(val_dataset, BATCH_SIZE, BUCKET_BOUNDARIES, shuffle=False)

train_loader = DataLoader(
    train_dataset,
    batch_sampler=train_sampler,
    num_workers=4,
    collate_fn=collate_fn,
    pin_memory=True
)
val_loader = DataLoader(
    val_dataset,
    batch_sampler=val_sampler,
    num_workers=4,
    collate_fn=collate_fn,
    pin_memory=True
)

model = ConformerCTC(
    vocab_size=VOCAB_SIZE,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    num_blocks=NUM_BLOCKS
)

train(model, train_loader, val_loader, resume=False)