import sys
from pathlib import Path

import torch
import train_config as cfg

# Allow imports from the experiment package root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmark_xfmr import TorchTransformer
from exp_helper import build_exp_for_train

 
def main():
    experiment_dir = Path(__file__).resolve().parent

    # Build the experiment so we reuse the exact same data/config setup.
    trainer, config = build_exp_for_train(
        experiment_dir,
        cfg.CONFIG,
        TorchTransformer
    )

    # Replace the model with the PyTorch Transformer implementation.
    model = TorchTransformer(
        src_vocab_size=config["src_vocab_size"],
        tgt_vocab_size=config["tgt_vocab_size"],
        d_model=config["d_model"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        d_ff=config["d_ff"],
        dropout=config["dropout"],
        src_pad_idx=config["src_pad_idx"],
        tgt_pad_idx=config["tgt_pad_idx"],
    )

    device = trainer.device
    model = model.to(device)

    checkpoint = experiment_dir / "checkpoints_eager" / "weights_final.pt"

    print(f"Checkpoint: {checkpoint}")
    print(f"Device:     {device}")
    print()

    state_dict = torch.load(
        checkpoint,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(state_dict)
    model.eval()

    print("=" * 70)
    print("TorchTransformer")
    print("=" * 70)
    print(model)
    print()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    print("=" * 70)
    print("Parameters")
    print("=" * 70)
    print(f"Total:     {total_params:,}")
    print(f"Trainable: {trainable_params:,}")
    print()

    print("=" * 70)
    print("Configuration")
    print("=" * 70)
    print(f"d_model:       {config['d_model']}")
    print(f"num_heads:     {config['num_heads']}")
    print(f"num_layers:    {config['num_layers']}")
    print(f"d_ff:          {config['d_ff']}")
    print(f"dropout:       {config['dropout']}")
    print()

    print("=" * 70)
    print("PyTorch nn.Transformer")
    print("=" * 70)

    transformer = model.transformer

    encoder_layer = transformer.encoder.layers[0]
    decoder_layer = transformer.decoder.layers[0]

    print(f"encoder norm_first:     {encoder_layer.norm_first}")
    print(f"decoder norm_first:     {decoder_layer.norm_first}")
    print(f"encoder activation:     {encoder_layer.activation}")
    print(f"decoder activation:     {decoder_layer.activation}")
    print(f"batch_first:            {encoder_layer.self_attn.batch_first}")
    print(f"encoder attention dropout: {encoder_layer.self_attn.dropout}")
    print(f"decoder self-attn dropout: {decoder_layer.self_attn.dropout}")
    print(f"decoder cross-attn dropout: {decoder_layer.multihead_attn.dropout}")
    print(f"layer norm eps:          {encoder_layer.norm1.eps}")
    print()

    print("=" * 70)
    print("Weight Tying")
    print("=" * 70)

    tied = (
        model.fc_out.weight.data_ptr()
        == model.decoder_embedding.weight.data_ptr()
    )

    print(f"decoder embedding ↔ output projection: {tied}")
    print()

    print("=" * 70)
    print("State Dict")
    print("=" * 70)

    for name, tensor in model.state_dict().items():
        print(f"{name:60s} {tuple(tensor.shape)}")


if __name__ == "__main__":
    main()