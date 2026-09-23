CONFIG = {
    # Execution
    "is_compile": False,

    # Model Implementation
    "model": "torch",

    # Data
    "data_dir": "data/multi30k/tok_wordlevel_data_build_hf",
    "raw_data_dir": "data/multi30k/raw",
    "train_src": "train_en_ids.pt",
    "train_tgt": "train_de_ids.pt",
    "val_src": "val_en_ids.pt",
    "val_tgt": "val_de_ids.pt",

    # Tokenizers
    "tokenizer_src": "tokenizer_en.json",
    "tokenizer_tgt": "tokenizer_de.json",

    # Special tokens
    "src_pad_token": "[PAD]",
    "tgt_pad_token": "[PAD]",

    # Training
    "batch_size": 32,
    "learning_rate": 1e-4,
    "epochs": 2,

    # # Model # for a local test run
    # "d_model": 128,
    # "num_heads": 8,
    # "num_layers": 6,
    # "d_ff": 512,
    # "dropout": 0.1,

    # Model # actual remote train
    "d_model": 512,
    "num_heads": 8,
    "num_layers": 6,
    "d_ff": 2048,
    "dropout": 0.1,

    # Optimizer
    "optimizer": "Adam",
    "betas": (0.9, 0.98),
    "eps": 1e-9,

    # Loss
    "label_smoothing": 0.1,
}