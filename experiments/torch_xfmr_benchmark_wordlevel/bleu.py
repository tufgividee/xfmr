import sys
from pathlib import Path

import sacrebleu
import train_config as cfg

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from data.multi30k.tok_wordlevel_data_build_hf.detokenizer import detokenize

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


def evaluate_bleu(source_file, target_file, 
                  max_lines: int | None = None):
    with open(source_file, encoding="utf-8") as f:
        sources = [line.strip() for line in f]

    with open(target_file, encoding="utf-8") as f:
        references = [line.strip().lower() for line in f]
        # we have been using lower cases for tokenization 2026 09 22

    if len(sources) != len(references):
        raise ValueError(
            f"Source/reference length mismatch: "
            f"{len(sources)} vs {len(references)}"
        )

    if max_lines is not None:
        sources = sources[:max_lines]
        references = references[:max_lines]

    hypotheses = []

    for i, (source, reference) in enumerate(zip(sources, references)):
        target = detokenize(translator.translate(source))
        hypotheses.append(target)

        if (i + 1) % 100 == 0:

            print("\nprompt:    " + source)
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
    max_lines=1000,  # limit to 1000 lines for training set
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