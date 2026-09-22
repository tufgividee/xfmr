import sys
from pathlib import Path

import sacrebleu
import train_config as cfg

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from exp_xfmr2017 import load_exp_for_infer

experiment_dir = Path(__file__).resolve().parent
project_dir = experiment_dir.parents[1]

checkpoint_name = "checkpoints_eager"
weights_name = "weights_final.pt"

translator, config = load_exp_for_infer(
    experiment_dir,
    checkpoint_name=checkpoint_name,
    weights_name=weights_name,
    is_compile=False,
)

data_dir = project_dir / cfg.CONFIG["raw_data_dir"]


import re


def detokenize(text):
    text = re.sub(r"\s+([?.!,;:])", r"\1", text)
    text = re.sub(r"([(\[{])\s+", r"\1", text)
    text = re.sub(r"\s+([)\]}])", r"\1", text)
    return text

def evaluate_bleu(source_file, target_file):
    with open(source_file, encoding="utf-8") as f:
        sources = [line.strip() for line in f]

    with open(target_file, encoding="utf-8") as f:
        references = [detokenize(line.strip()) for line in f]

    if len(sources) != len(references):
        raise ValueError(
            f"Source/reference length mismatch: "
            f"{len(sources)} vs {len(references)}"
        )

    hypotheses = []

    for i, (source, reference) in enumerate(zip(sources, references)):
        reference = detokenize(reference).lower()
        target = detokenize(translator.translate(source)).lower()
        hypotheses.append(target)

        if (i + 1) % 100 == 0:
            print("prompt:    " + source)
            print("reference: " + reference)
            print("response:  " + target)
            print(f"Translated {i + 1}/{len(sources)}")



    return sacrebleu.corpus_bleu(
        hypotheses,
        [references],
    )


train_bleu = evaluate_bleu(
    data_dir / "train.en",
    data_dir / "train.de",
)

print(f"\nTrain BLEU: {train_bleu.score:.2f}")

val_bleu = evaluate_bleu(
    data_dir / "val.en",
    data_dir / "val.de",
)

print(f"\nValidation BLEU: {val_bleu.score:.2f}")


test_bleu = evaluate_bleu(
    data_dir / "test_2016_flickr.en",
    data_dir / "test_2016_flickr.de",
)

print(f"Test BLEU:       {test_bleu.score:.2f}")



# with open(experiment_dir / "bleu.log", "a", encoding="utf-8") as g:
#     g.write(f"dataset: {data_dir}\n")
#     g.write(f"checkpoint: {checkpoint_name}/{weights_name}\n")
#     g.write(f"Validation BLEU: {val_bleu.score:.2f}\n")
#     g.write(f"Test BLEU:       {test_bleu.score:.2f}\n")
#     g.write("\n")