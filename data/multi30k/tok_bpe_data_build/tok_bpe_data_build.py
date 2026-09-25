from pathlib import Path

import torch
from tokenizers import Tokenizer
from tokenizers.decoders import WordPiece
from tokenizers.models import BPE
from tokenizers.normalizers import Lowercase
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer

# Setup directory:
# data/multi30k/tok_bpe_data_build
SCRIPT_DIR = Path(__file__).resolve().parent

# Raw data directory:
# data/multi30k/raw
RAW_DIR = SCRIPT_DIR.parent / "raw"


def setup_tokenizer() -> Tokenizer:
    tokenizer = Tokenizer(
        BPE(
            unk_token="[UNK]",
            continuing_subword_prefix="##",
        )
    )
    tokenizer.normalizer = Lowercase()
    tokenizer.pre_tokenizer = Whitespace()
    # tokenizer.decoder = BPEDecoder(suffix="##")
    tokenizer.decoder = WordPiece(prefix="##")
    # strickly we are not using BPE but a hybrid for now
    return tokenizer


def add_post_processor(tok: Tokenizer) -> None:
    tok.post_processor = TemplateProcessing(
        single="[SOS] $A [EOS]",
        special_tokens=[
            ("[SOS]", tok.token_to_id("[SOS]")),
            ("[EOS]", tok.token_to_id("[EOS]")),
        ],
    )


def get_lines(file_path: Path):
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            yield line.strip()


def get_training_lines():
    """Stream English and German training data into one shared vocabulary."""
    yield from get_lines(RAW_DIR / "train.en")
    yield from get_lines(RAW_DIR / "train.de")


def make_tokenizer() -> Tokenizer:
    tokenizer = setup_tokenizer()

    trainer = BpeTrainer(
        special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"],
        vocab_size=8000,
        min_frequency=2,
        continuing_subword_prefix="##",
    ) # strickly we are not using BPE but a hybrid for now

    print("Training shared BPE vocabulary on train.en + train.de...")

    tokenizer.train_from_iterator(
        get_training_lines(),
        trainer,
    )

    add_post_processor(tokenizer)

    save_path = SCRIPT_DIR / "tokenizer.json"
    tokenizer.save(str(save_path))

    print(f"Saved tokenizer -> {save_path.name}")
    print(f"Vocabulary size: {tokenizer.get_vocab_size()}")

    return tokenizer


def make_dataset(
    raw_file_name: str,
    dataset_name: str,
    tokenizer: Tokenizer,
) -> None:
    print(f"Encoding {raw_file_name}...")

    dataset = []

    with open(RAW_DIR / raw_file_name, "r", encoding="utf-8") as f:
        for line in f:
            dataset.append(
                tokenizer.encode(line.strip()).ids
            )

    save_path = SCRIPT_DIR / f"{dataset_name}.pt"
    torch.save(dataset, save_path)

    print(
        f"Saved split -> {save_path.name} "
        f"({len(dataset)} items)"
    )


if __name__ == "__main__":
    print("Making tokenizer and datasets for Multi30k (BPE)...")

    # 1. Train ONE shared tokenizer using training data only.
    tokenizer = make_tokenizer()

    # 2. Encode all splits using the same tokenizer.
    splits = {
        "train": ("train.en", "train.de"),
        "val": ("val.en", "val.de"),
        "test": ("test_2016_flickr.en", "test_2016_flickr.de"),
    }

    for split_name, (en_file, de_file) in splits.items():
        make_dataset(
            en_file,
            f"{split_name}_en_ids",
            tokenizer,
        )

        make_dataset(
            de_file,
            f"{split_name}_de_ids",
            tokenizer,
        )

    print("BPE data preparation complete!")