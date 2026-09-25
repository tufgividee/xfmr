import json
from pathlib import Path

import torch
from torch import nn, optim

from xfmr.xfmr2017.data import create_dataloader, load_tokenizer
from xfmr.xfmr2017.inference import Translator
from xfmr.xfmr2017.training import Trainer


def build_exp_for_train(
    experiment_dir: Path,
    config,
    model_class,
    ):

    checkpoint_dir = experiment_dir / (
        "checkpoints_compile"
        if config["is_compile"]
        else "checkpoints_eager"
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    project_dir = experiment_dir.parents[1]
    data_dir = project_dir / config["data_dir"]

    # Tokenizers
    tok_src = load_tokenizer(
        data_dir / config["tokenizer_src"]
    )
    tok_tgt = load_tokenizer(
        data_dir / config["tokenizer_tgt"]
    )

    # Derived tokenizer values
    src_vocab_size = tok_src.get_vocab_size()
    tgt_vocab_size = tok_tgt.get_vocab_size()

    src_pad_idx = tok_src.token_to_id(
        config["src_pad_token"]
    )
    tgt_pad_idx = tok_tgt.token_to_id(
        config["tgt_pad_token"]
    )

    # Data
    train_loader = create_dataloader(
        files=[
            data_dir / config["train_src"],
            data_dir / config["train_tgt"],
        ],
        pad_ids=[src_pad_idx, tgt_pad_idx],
        batch_size=config["batch_size"],
        shuffle=True,
    )

    val_loader = create_dataloader(
        files=[
            data_dir / config["val_src"],
            data_dir / config["val_tgt"],
        ],
        pad_ids=[src_pad_idx, tgt_pad_idx],
        batch_size=config["batch_size"],
        shuffle=False,
    )

    # Device
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = model_class(
        src_vocab_size,
        tgt_vocab_size,
        config["d_model"],
        config["num_heads"],
        config["num_layers"],
        config["d_ff"],
        config["dropout"],
        src_pad_idx=src_pad_idx,
        tgt_pad_idx=tgt_pad_idx,
    ).to(device)

    # Loss
    criterion = nn.CrossEntropyLoss(
        ignore_index=tgt_pad_idx,
        label_smoothing=config["label_smoothing"],
    )

    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        betas=config["betas"],
        eps=config["eps"],
    )

    # Resolved experiment configuration
    resolved_config = {
        **config,
        "dataset": str(data_dir.relative_to(project_dir)),
        "src_vocab_size": src_vocab_size,
        "tgt_vocab_size": tgt_vocab_size,
        "src_pad_idx": src_pad_idx,
        "tgt_pad_idx": tgt_pad_idx,
        "torch_compile": config["is_compile"],
    }
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        checkpoint_dir=checkpoint_dir,
        config=resolved_config,
        device=device,
    )

    return trainer, resolved_config


def load_exp_for_infer(
    experiment_dir: Path,
    checkpoint_name: str,
    weights_name: str,
    is_compile: bool,
    model_class,
    ):

    checkpoint_dir = experiment_dir / checkpoint_name
    project_dir = experiment_dir.parents[1]

    with open(checkpoint_dir / "config.json") as f:
        config = json.load(f)

    data_dir = project_dir / config["dataset"]

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = model_class(
        config["src_vocab_size"],
        config["tgt_vocab_size"],
        config["d_model"],
        config["num_heads"],
        config["num_layers"],
        config["d_ff"],
        config["dropout"],
        src_pad_idx=config["src_pad_idx"],
        tgt_pad_idx=config["tgt_pad_idx"],
    ).to(device)

    model.load_state_dict(
        torch.load(
            checkpoint_dir / weights_name,
            map_location=device,
            weights_only=True,
        )
    )

    tok_src = load_tokenizer(
        data_dir / config["tokenizer_src"]
    )
    tok_tgt = load_tokenizer(
        data_dir / config["tokenizer_tgt"]
    )

    if is_compile:
        model = torch.compile(model)

    translator = Translator(
        model=model,
        src_tokenizer=tok_src,
        tgt_tokenizer=tok_tgt,
        device=device,
    )

    return translator, config


# from transformers import get_linear_schedule_with_warmup

# # Setup parameters
# num_epochs = 10
# total_steps = len(train_loader) * num_epochs
# warmup_steps = int(total_steps * 0.1)  # 10% warmup

# optimizer = build_bert_optimizer(model, lr=1e-4, weight_decay=0.01)

# scheduler = get_linear_schedule_with_warmup(
#     optimizer,
#     num_warmup_steps=warmup_steps,
#     num_training_steps=total_steps,
# )

# trainer = Trainer(
#     model=model,
#     train_loader=train_loader,
#     val_loader=val_loader,
#     mlm_criterion=nn.CrossEntropyLoss(ignore_index=-100),
#     nsp_criterion=nn.CrossEntropyLoss(),
#     optimizer=optimizer,
#     scheduler=scheduler,
#     checkpoint_dir="./checkpoints_bert",
#     config={"torch_compile": True, "batch_size": 32},
#     amp_dtype=torch.bfloat16,  # Use torch.bfloat16 or torch.float16
# )

# trainer.fit(epochs=num_epochs)