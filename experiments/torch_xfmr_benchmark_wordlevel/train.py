import sys
from pathlib import Path

import train_config as cfg

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmark_xfmr import TorchTransformer
from exp_helper import build_exp_for_train

experiment_dir = Path(__file__).resolve().parent

trainer, config = build_exp_for_train(
    experiment_dir=experiment_dir,
    config=cfg.CONFIG,
    model_class=TorchTransformer,
)

trainer.fit(epochs=cfg.CONFIG["epochs"])