from pathlib import Path

import torch
from tokenizers import Tokenizer as HFTokenizer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


def load_tokenizer(tokenizer_path: str | Path) -> HFTokenizer:
    """Helper to load an HF Tokenizer safely from a string or Path object."""
    return HFTokenizer.from_file(str(tokenizer_path))


class SingleDataset(Dataset):
    """Loads a single tokenized .pt file containing token IDs."""

    def __init__(self, pt_path: str | Path):
        self.ids = torch.load(Path(pt_path), weights_only=True)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.tensor(self.ids[idx], dtype=torch.long)


class AlignedDatasets(Dataset):
    """Combines N single datasets and enforces index synchronization."""

    def __init__(self, *datasets: Dataset):
        if not datasets:
            raise ValueError("At least one dataset must be provided.")
        first_len = len(datasets[0])
        if not all(len(ds) == first_len for ds in datasets):
            lengths = [len(ds) for ds in datasets]
            raise ValueError(f"Dataset length mismatch across streams: {lengths}")
        self.datasets = datasets

    def __len__(self) -> int:
        return len(self.datasets[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
        return tuple(ds[idx] for ds in self.datasets)


class Collate:
    """Pads N streams independently to their respective max lengths in each batch."""

    def __init__(self, pad_ids: list[int]):
        self.pad_ids = pad_ids

    def __call__(
        self, batch: list[tuple[torch.Tensor, ...]]
    ) -> tuple[torch.Tensor, ...]:
        streams = zip(*batch)  # Transpose batch from list-of-tuples to tuple-of-lists
        return tuple(
            pad_sequence(stream, batch_first=True, padding_value=pad_id)
            for stream, pad_id in zip(streams, self.pad_ids)
        )


def create_dataloader(
    files: list[str | Path],
    pad_ids: list[int],
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Factory function that loads and pairs any N streams into a single DataLoader."""
    if len(files) != len(pad_ids):
        raise ValueError(
            f"Number of files ({len(files)}) must match number of pad_ids ({len(pad_ids)})."
        )

    single_datasets = [SingleDataset(f) for f in files]
    aligned_ds = AlignedDatasets(*single_datasets)


    collate_fn = Collate(pad_ids=pad_ids)

    return DataLoader(
        corrupted_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )


# =====================================================================
# Verification Script
# =====================================================================
if __name__ == "__main__":
    SCRIPT_DIR = Path(__file__).resolve().parent
    DATA_DIR = SCRIPT_DIR.parent / "data/multi30k/tok_wordlevel_data_buid_hf"

    # Load native tokenizers using helper
    tok_en = load_tokenizer(DATA_DIR / "tokenizer_en.json")
    tok_de = load_tokenizer(DATA_DIR / "tokenizer_de.json")

    pad_en = tok_en.token_to_id("[PAD]")
    pad_de = tok_de.token_to_id("[PAD]")

    # N = 2 (Seq2Seq / Machine Translation)
    train_loader_nmt = create_dataloader(
        files=[DATA_DIR / "train_de_ids.pt", DATA_DIR / "train_en_ids.pt"],
        pad_ids=[pad_de, pad_en],
        batch_size=32,
        shuffle=True,
    )

    for batch_de, batch_en in train_loader_nmt:
        print("N=2 Shapes (DE, EN):", batch_de.shape, batch_en.shape)
        # break

    # N = 1 (Language Modeling)
    train_loader_lm = create_dataloader(
        files=[DATA_DIR / "train_en_ids.pt"],
        pad_ids=[pad_en],
        batch_size=32,
        shuffle=True,
    )

    for (batch_en,) in train_loader_lm:
        print("N=1 Shape (EN):", batch_en.shape)
        # break




def load_tokenizer(tokenizer_path: str | Path) -> HFTokenizer:
    """Load a Hugging Face tokenizer from a JSON file."""
    return HFTokenizer.from_file(str(tokenizer_path))


def load_documents(pt_path: str | Path) -> list[list[list[int]]]:
    """Load tokenized documents from a .pt file."""
    return torch.load(
        Path(pt_path),
        weights_only=True,
    )


def create_dataloader(
    files: list[str | Path],
    pad_ids: list[int],
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Factory function that loads and pairs any N streams into a single DataLoader."""
    if len(files) != len(pad_ids):
        raise ValueError(
            f"Number of files ({len(files)}) must match number of pad_ids ({len(pad_ids)})."
        )

    single_datasets = [SingleDataset(f) for f in files]
    aligned_ds = AlignedDatasets(*single_datasets)
    collate_fn = MultiStreamPadCollate(pad_ids=pad_ids)

    return DataLoader(
        aligned_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )