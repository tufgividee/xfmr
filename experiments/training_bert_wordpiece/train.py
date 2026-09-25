import sys
from pathlib import Path

import train_config as cfg

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from exp_helper import build_exp_for_train

from xfmr.bert2018.bert import BertModel

experiment_dir = Path(__file__).resolve().parent

trainer, config = build_exp_for_train(
    experiment_dir=experiment_dir,
    config=cfg.CONFIG,
    model_class=BertModel,
)

trainer.fit(epochs=cfg.CONFIG["epochs"])