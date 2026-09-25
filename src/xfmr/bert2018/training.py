import datetime
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch import nn, optim
from torch.utils.data import DataLoader


def build_bert_optimizer(
    model: nn.Module,
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    betas: Tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-6,
) -> optim.AdamW:
    """Configures AdamW with weight decay exclusion for biases and LayerNorms."""
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay) and p.requires_grad
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay) and p.requires_grad
            ],
            "weight_decay": 0.0,
        },
    ]
    return optim.AdamW(optimizer_grouped_parameters, lr=lr, betas=betas, eps=eps)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        mlm_criterion: nn.Module,
        nsp_criterion: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Optional[Any],
        checkpoint_dir: Union[str, Path],
        config: Dict[str, Any],
        device: Optional[torch.device] = None,
        use_amp: bool = True,
        amp_dtype: torch.dtype = torch.bfloat16,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config
        self.use_amp = use_amp and self.device.type == "cuda"
        self.amp_dtype = amp_dtype

        # Model compilation & placement
        self.raw_model = model.to(self.device)
        if self.config.get("torch_compile", False):
            self.model = torch.compile(self.raw_model)
        else:
            self.model = self.raw_model

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.mlm_criterion = mlm_criterion
        self.nsp_criterion = nsp_criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

        # Gradient Scaler for FP16 (bfloat16 does not require scaling)
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=(self.use_amp and self.amp_dtype == torch.float16)
        )

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.config_path = self.checkpoint_dir / "config.json"
        self.log_path = self.checkpoint_dir / "training.log"
        self.log_path.unlink(missing_ok=True)

    def save_config(self, timestamp_start: str) -> None:
        try:
            git_commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip()
            git_dirty = bool(
                subprocess.check_output(
                    ["git", "status", "--porcelain"], text=True
                ).strip()
            )
        except Exception:
            git_commit, git_dirty = "N/A", False

        run_config = {
            **self.config,
            "timestamp_start": timestamp_start,
            "device": str(self.device),
            "pytorch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "git_commit": git_commit,
            "git_dirty": git_dirty,
            "amp_enabled": self.use_amp,
            "amp_dtype": str(self.amp_dtype),
        }

        if self.device.type == "cuda":
            run_config["gpu"] = torch.cuda.get_device_name(self.device)

        with open(self.config_path, "w") as f:
            json.dump(run_config, f, indent=4)

    def log(self, message: str) -> None:
        print(message)
        with open(self.log_path, "a") as f:
            f.write(message + "\n")

    def _run_epoch(self, training: bool) -> Tuple[float, float]:
        loader = self.train_loader if training else self.val_loader
        self.model.train(training)

        total_loss = 0.0
        total_samples = 0
        start_time = time.perf_counter()

        context = torch.enable_grad() if training else torch.no_grad()

        with context:
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device, non_blocking=True)
                token_type_ids = batch["token_type_ids"].to(self.device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)
                mlm_labels = batch["mlm_labels"].to(self.device, non_blocking=True)
                nsp_label = batch["nsp_label"].to(self.device, non_blocking=True)

                batch_size = input_ids.size(0)

                if training:
                    self.optimizer.zero_grad(set_to_none=True)

                # Mixed Precision Forward Pass
                with torch.amp.autocast(
                    device_type=self.device.type,
                    enabled=self.use_amp,
                    dtype=self.amp_dtype,
                ):
                    mlm_logits, nsp_logits = self.model(
                        input_ids,
                        token_type_ids,
                        attention_mask,
                    )

                    mlm_loss = self.mlm_criterion(
                        mlm_logits.view(-1, mlm_logits.size(-1)),
                        mlm_labels.view(-1),
                    )
                    nsp_loss = self.nsp_criterion(
                        nsp_logits,
                        nsp_label,
                    )
                    loss = mlm_loss + nsp_loss

                if training:
                    # Scaled Backwards Pass
                    self.scaler.scale(loss).backward()

                    # Unscale gradients for clipping
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=1.0
                    )

                    # Scaler Step
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                    if self.scheduler is not None:
                        self.scheduler.step()

                total_loss += loss.item() * batch_size
                total_samples += batch_size

        elapsed = time.perf_counter() - start_time
        throughput = total_samples / elapsed
        avg_loss = total_loss / total_samples

        return avg_loss, throughput

    def train_epoch(self) -> Tuple[float, float]:
        return self._run_epoch(training=True)

    def validate(self) -> Tuple[float, float]:
        return self._run_epoch(training=False)

    def reset_peak_gpu_memory(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

    def peak_gpu_memory_mb(self) -> Optional[float]:
        if self.device.type != "cuda":
            return None
        return torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)

    def fit(self, epochs: int) -> None:
        self.reset_peak_gpu_memory()

        timestamp_start = datetime.datetime.now(datetime.timezone.utc).isoformat()
        start_time = time.perf_counter()

        self.save_config(timestamp_start)
        self.log(f"Launching training on: {self.device}")
        self.log(f"Training started: {timestamp_start}")

        best_val_loss = float("inf")

        for epoch in range(1, epochs + 1):
            epoch_start = time.perf_counter()

            train_loss, train_throughput = self.train_epoch()
            val_loss, val_throughput = self.validate()

            epoch_time = time.perf_counter() - epoch_start
            current_lr = self.optimizer.param_groups[0]["lr"]

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save(
                    self.checkpoint_dir / "weights_min_val_loss.pt", epoch, val_loss
                )
                self.log(
                    f"--> Best validation loss improved to {best_val_loss:.4f} at epoch {epoch}"
                )

            self.log(
                f"Epoch {epoch:02d}/{epochs:02d} | "
                f"Train Loss: {train_loss:.4f} ({train_throughput:.1f} samples/s) | "
                f"Val Loss: {val_loss:.4f} ({val_throughput:.1f} samples/s) | "
                f"LR: {current_lr:.2e} | Time: {epoch_time:.2f}s"
            )

        total_time = time.perf_counter() - start_time
        timestamp_end = datetime.datetime.now(datetime.timezone.utc).isoformat()
        self.save(self.checkpoint_dir / "weights_final.pt", epochs, val_loss)

        self.log(f"Training completed: {timestamp_end}")
        self.log(f"Total training time: {total_time:.2f} s")
        self.log(f"Average epoch time: {total_time / epochs:.2f} s")

        peak_memory = self.peak_gpu_memory_mb()
        if peak_memory is not None:
            self.log(f"Peak GPU memory usage: {peak_memory:.2f} MB")

    def save(self, path: Union[str, Path], epoch: int, val_loss: float) -> None:
        """Saves a checkpoint containing raw model parameters, optimizer, and scheduler states."""
        # Retrieve non-compiled base module
        model_to_save = getattr(self.model, "_orig_mod", self.model)

        checkpoint = {
            "epoch": epoch,
            "val_loss": val_loss,
            "model_state_dict": model_to_save.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
        }

        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        if self.use_amp and self.amp_dtype == torch.float16:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        torch.save(checkpoint, path)