CONFIG = {
    # Execution
    "is_compile": False,
    "model": "xfmr2017",

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
    "epochs": 50,

    # Model
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

CONFIG = {
    # Data paths
    "data_dir": "data/wikitext2/tok_wordpiece_bert_data_build",
    "tokenizer": "tokenizer.json",
    "train_ids": "train_ids.pt",
    "val_ids": "val_ids.pt",
    
    # Sequence & Batch settings
    "max_seq_len": 128,
    "batch_size": 32,
    "num_workers": 2,
    
    # Model Specs (BERT-Base or Small)
    "d_model": 256,
    "num_heads": 8,
    "num_layers": 6,
    "d_ff": 1024,
    "dropout": 0.1,
    
    # Optimization
    "epochs": 10,
    "learning_rate": 1e-4,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    
    # Compilation & Precision
    "is_compile": True,
    "use_amp": True,
}