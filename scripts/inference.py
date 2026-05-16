import torch
from torch.utils.data import DataLoader
from hindi_asr.model import ConformerCTC as ASRModel
from hindi_asr.dataset import HindiASRDataset, collate_fn, MANIFEST_PATH_VAL, STATS_PATH
from hindi_asr.decoder import greedy_decode, load_vocab
from hindi_asr.checkpoints import load_checkpoint

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

VOCAB_PATH = "data/vocab.json"
CHECKPOINT_PATH = "checkpoints/best.pt"

BATCH_SIZE = 8
NUM_SAMPLES = 20

def run_inference():

    vocab_dict, vocab_dict_rev = load_vocab(VOCAB_PATH)

    blank_id = vocab_dict["<blank>"]

    dataset = HindiASRDataset(
    MANIFEST_PATH_VAL,
    STATS_PATH,
    is_train=False
)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )

    model = ASRModel(
        vocab_size=len(vocab_dict)
    )

    load_checkpoint(
        CHECKPOINT_PATH,
        model,
        optimizer=None,
        scheduler=None
    )

    model.to(DEVICE)
    model.eval()

    printed = 0

    with torch.no_grad():

        for batch in loader:

            mels, tokens, mel_lengths, token_lengths = batch

            mels = mels.to(DEVICE, non_blocking=True)
            mel_lengths = mel_lengths.to(DEVICE, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):

                log_probs, input_lengths = model(
                    mels,
                    mel_lengths
                )

            hyps = greedy_decode(
                log_probs.cpu(),
                input_lengths.cpu(),
                vocab_dict_rev,
                blank_id
            )

            for i, hyp in enumerate(hyps):

                ref_tokens = tokens[i, :token_lengths[i]].tolist()

                ref = "".join([
                    vocab_dict_rev.get(t, "<unk>")
                    for t in ref_tokens
                ])

                print(f"\nSample {printed + 1}")
                print(f"REF : {ref}")
                print(f"HYP : {hyp}")

                printed += 1

                if printed >= NUM_SAMPLES:
                    return


if __name__ == "__main__":
    run_inference()