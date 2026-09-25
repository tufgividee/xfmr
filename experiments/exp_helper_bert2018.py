import json
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup

from xfmr.bert2018.data import (
    SentencePairSampler,
    load_documents,
    load_tokenizer,
    make_batch,
)
from xfmr.bert2018.training import Trainer


class BERTPretrainingDataset(Dataset):
    """
    Adapter dataset wrapping SentencePairSampler to make it compatible
    with PyTorch DataLoader and multi-worker batch generation.
    """
    def __init__(self, documents: list[list[list[int]]], max_seq_len: int = 128):
        self.sampler = SentencePairSampler(documents=documents, max_seq_len=max_seq_len)

    def __len__(self) -> int:
        return self.sampler.n_sample

    def __getitem__(self, idx: int):
        # Fallback element accessor if needed by index
        sample = self.sampler.sample_schedule[idx]
        doc_a, seg_a, doc_b, seg_b, nsp_label = sample
        
        start_a, end_a = self.sampler.segment_table[doc_a][seg_a]
        start_b, end_b = self.sampler.segment_table[doc_b][seg_b]

        tokens_a = [
            token
            for sentence in self.sampler.documents[doc_a][start_a:end_a]
            for token in sentence
        ]
        tokens_b = [
            token
            for sentence in self.sampler.documents[doc_b][start_b:end_b]
            for token in sentence
        ]

        return tokens_a, tokens_b, nsp_label


def build_bert_optimizer(model: nn.Module, lr: float = 1e-4, weight_decay: float = 0.01):
    """Configures AdamW with weight decay exclusion for bias and LayerNorm parameters."""
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    return optim.AdamW(optimizer_grouped_parameters, lr=lr, betas=(0.9, 0.999), eps=1e-6)


def build_exp_for_train(
    experiment_dir: Path,
    config: dict,
    model_class,
):
    checkpoint_dir = experiment_dir / (
        "checkpoints_compile" if config.get("is_compile", False) else "checkpoints_eager"
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    project_dir = experiment_dir.parents[1]
    data_dir = project_dir / config["data_dir"]

    # Load Tokenizer
    tokenizer = load_tokenizer(data_dir / config["tokenizer"])
    vocab_size = tokenizer.get_vocab_size()

    cls_token_id = tokenizer.token_to_id("[CLS]")
    sep_token_id = tokenizer.token_to_id("[SEP]")
    mask_token_id = tokenizer.token_to_id("[MASK]")
    pad_token_id = tokenizer.token_to_id("[PAD]")

    # Load Data
    train_docs = load_documents(data_dir / config["train_ids"])
    val_docs = load_documents(data_dir / config["val_ids"])

    train_dataset = BERTPretrainingDataset(train_docs, max_seq_len=config["max_seq_len"])
    val_dataset = BERTPretrainingDataset(val_docs, max_seq_len=config["max_seq_len"])

    def collate_fn(samples):
        return make_batch(
            samples=samples,
            vocab_size=vocab_size,
            cls_token_id=cls_token_id,
            sep_token_id=sep_token_id,
            mask_token_id=mask_token_id,
            pad_token_id=pad_token_id,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config.get("num_workers", 0),
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config.get("num_workers", 0),
        pin_memory=True,
    )

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate BERT Model
    model = model_class(
        vocab_size=vocab_size,
        d_model=config["d_model"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        d_ff=config["d_ff"],
        dropout=config["dropout"],
        max_seq_len=config["max_seq_len"],
        pad_idx=pad_token_id,
    ).to(device)

    if config.get("is_compile", False):
        model = torch.compile(model)

    # Criterions
    mlm_criterion = nn.CrossEntropyLoss(ignore_index=-100)
    nsp_criterion = nn.CrossEntropyLoss()

    # Optimizer & LR Scheduler
    optimizer = build_bert_optimizer(
        model,
        lr=config["learning_rate"],
        weight_decay=config.get("weight_decay", 0.01),
    )

    total_steps = len(train_loader) * config["epochs"]
    warmup_steps = int(total_steps * config.get("warmup_ratio", 0.1))

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # Resolved Experiment Configuration
    resolved_config = {
        **config,
        "vocab_size": vocab_size,
        "cls_token_id": cls_token_id,
        "sep_token_id": sep_token_id,
        "mask_token_id": mask_token_id,
        "pad_token_id": pad_token_id,
        "total_steps": total_steps,
        "warmup_steps": warmup_steps,
    }

    # Save resolved config for checkpointing
    with open(checkpoint_dir / "config.json", "w") as f:
        json.dump(resolved_config, f, indent=2)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        mlm_criterion=mlm_criterion,
        nsp_criterion=nsp_criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        checkpoint_dir=checkpoint_dir,
        config=resolved_config,
        device=device,
        amp_dtype=torch.bfloat16 if config.get("use_amp", False) else None,
    )

    return trainer, resolved_config