import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from data.multi30k.tok_wordlevel_data_build_hf.detokenizer import detokenize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from exp_helper import load_exp_for_infer

from xfmr.xfmr2017.transformer import Transformer

experiment_dir = Path(__file__).resolve().parent

translator, config = load_exp_for_infer(
    experiment_dir = experiment_dir,
    checkpoint_name="checkpoints_eager",
    weights_name="weights_final.pt",
    is_compile=False,
    model_class=Transformer,
)

prompt = "Two young, White males are outside near many bushes."

response = translator.translate(prompt)

print(f"\nSource: {prompt}")
print(f"Translation: {detokenize(response)}")