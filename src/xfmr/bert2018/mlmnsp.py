import random

import torch

from xfmr.bert2018.data import load_tokenizer


class SentencePairSampler:
    def __init__(
        self,
        documents: list[list[list[int]]],
        max_seq_len: int = 128,
    ):
        self.documents = documents
        self.max_pair_tokens = max_seq_len - 3

        self.document_index = 0
        self.sentence_index = 0

        self.tok_count_table = self.make_tok_count_table()
        self.segment_table = self.make_segment_table()
        self.sample_schedule = self.make_sample_schedule()
        self.n_sample = len(self.sample_schedule)
        self.i_sample = 0

    def make_tok_count_table(self):
        return [
            [len(sentence) for sentence in document]
            for document in self.documents
        ]         

    # The paper did not describe how data segmentation is done,
    # so we assume:
    # 1) pack sequences close to max_seq_len, and
    # 2) maintain a balance between A and B segments.
    #
    # Therefore:
    # - each document is divided into contiguous segments
    # - each segment is roughly half of the available token budget
    # - adjacent segments form an IsNext pair
    # - and the first segment of a document is A

    def make_segment_table(self):
        segment_target = self.max_pair_tokens // 2
        segment_table = []

        for document in self.tok_count_table:
            segments = []
            start = 0
            token_count = 0

            for sentence_idx, sentence_len in enumerate(document):
                if (token_count + sentence_len > segment_target 
                    and token_count > 0):

                    segments.append((start, sentence_idx))
                    start = sentence_idx
                    token_count = 0

                token_count += sentence_len

            if start < len(document):
                segments.append((start, len(document)))

            segment_table.append(segments)

        return segment_table

    def make_sample_schedule(self):
        choices = [
            [random.randint(0, 1) for _ in segments]
            for segments in self.segment_table
        ]

        schedule = []

        for doc_a, segments in enumerate(self.segment_table):
            for seg_a, _ in enumerate(segments):
                nsp_label = choices[doc_a][seg_a]

                if nsp_label == 1:
                    # IsNext
                    if seg_a + 1 >= len(segments):
                        continue

                    doc_b = doc_a
                    seg_b = seg_a + 1

                else:
                    # NotNext: choose any segment that is not the
                    # actual next segment.
                    candidates = [
                        (doc_idx, seg_idx)
                        for doc_idx, doc_segments in enumerate(self.segment_table)
                        for seg_idx in range(len(doc_segments))
                        if not (
                            doc_idx == doc_a
                            and seg_idx == seg_a + 1
                        )
                    ]

                    doc_b, seg_b = random.choice(candidates)

                schedule.append(
                    (doc_a, seg_a, doc_b, seg_b, nsp_label)
                )

        return schedule

    def sample(self, batch_size):
        start = self.i_sample
        end = min(start + batch_size, self.n_sample)

        batch = []

        for doc_a, seg_a, doc_b, seg_b, nsp_label in self.sample_schedule[start:end]:
            start_a, end_a = self.segment_table[doc_a][seg_a]
            start_b, end_b = self.segment_table[doc_b][seg_b]

            tokens_a = [
                token
                for sentence in self.documents[doc_a][start_a:end_a]
                for token in sentence
            ]

            tokens_b = [
                token
                for sentence in self.documents[doc_b][start_b:end_b]
                for token in sentence
            ]

            batch.append((tokens_a, tokens_b, nsp_label))

        self.i_sample = end

        return batch


def apply_mlm_batch(
    input_ids: torch.Tensor,
    vocab_size: int,
    mask_token_id: int,
    special_token_ids: set[int],
    mlm_probability: float = 0.15,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply BERT's 80/10/10 MLM corruption to a padded batch.

    Args:
        input_ids: [batch, seq_len] padded token IDs.

    Returns:
        corrupted input_ids
        mlm_labels with original token IDs at selected positions
        and -100 elsewhere.
    """

    mlm_labels = torch.full_like(
        input_ids,
        -100,
    )

    eligible = torch.ones_like(
        input_ids,
        dtype=torch.bool,
    )

    for token_id in special_token_ids:
        eligible &= input_ids != token_id

    selected = (
        torch.rand(
            input_ids.shape,
            device=input_ids.device,
        )
        < mlm_probability
    )

    selected &= eligible

    mlm_labels[selected] = input_ids[selected]

    corruption = torch.rand(
        input_ids.shape,
        device=input_ids.device,
    )

    mask_positions = selected & (corruption < 0.80)

    random_positions = selected & (
        (corruption >= 0.80)
        & (corruption < 0.90)
    )

    input_ids = input_ids.clone()

    input_ids[mask_positions] = mask_token_id

    input_ids[random_positions] = torch.randint(
        0,
        vocab_size,
        size=(random_positions.sum().item(),),
        device=input_ids.device,
    )

    return input_ids, mlm_labels


def make_batch(
    samples: list[tuple[list[int], list[int], int]],
    vocab_size: int,
    cls_token_id: int,
    sep_token_id: int,
    mask_token_id: int,
    pad_token_id: int,
) -> dict[str, torch.Tensor]:

    examples = []

    for tokens_a, tokens_b, nsp_label in samples:
        input_ids = (
            [cls_token_id]
            + tokens_a
            + [sep_token_id]
            + tokens_b
            + [sep_token_id]
        )

        token_type_ids = (
            [0] * (len(tokens_a) + 2)
            + [1] * (len(tokens_b) + 1)
        )

        examples.append(
            {
                "input_ids": torch.tensor(
                    input_ids,
                    dtype=torch.long,
                ),
                "token_type_ids": torch.tensor(
                    token_type_ids,
                    dtype=torch.long,
                ),
                "nsp_label": nsp_label,
            }
        )

    max_len = max(
        example["input_ids"].size(0)
        for example in examples
    )

    batch_size = len(examples)

    input_ids = torch.full(
        (batch_size, max_len),
        pad_token_id,
        dtype=torch.long,
    )

    token_type_ids = torch.zeros(
        (batch_size, max_len),
        dtype=torch.long,
    )

    attention_mask = torch.zeros(
        (batch_size, max_len),
        dtype=torch.long,
    )

    nsp_labels = torch.empty(
        batch_size,
        dtype=torch.long,
    )

    for i, example in enumerate(examples):
        seq_len = example["input_ids"].size(0)

        input_ids[i, :seq_len] = example["input_ids"]
        token_type_ids[i, :seq_len] = example["token_type_ids"]
        attention_mask[i, :seq_len] = 1
        nsp_labels[i] = example["nsp_label"]

    input_ids, mlm_labels = apply_mlm_batch(
        input_ids=input_ids,
        vocab_size=vocab_size,
        mask_token_id=mask_token_id,
        special_token_ids={
            cls_token_id,
            sep_token_id,
            pad_token_id,
        },
    )

    return {
        "input_ids": input_ids,
        "token_type_ids": token_type_ids,
        "attention_mask": attention_mask,
        "mlm_labels": mlm_labels,
        "nsp_label": nsp_labels,
    }


def make_example(
    tokens_a: list[int],
    tokens_b: list[int],
    nsp_label: int,
    vocab_size: int,
    cls_token_id: int,
    sep_token_id: int,
    mask_token_id: int,
    pad_token_id: int,
) -> dict[str, torch.Tensor]:

    input_ids = (
        [cls_token_id]
        + tokens_a
        + [sep_token_id]
        + tokens_b
        + [sep_token_id]
    )

    token_type_ids = (
        [0] * (len(tokens_a) + 2)
        + [1] * (len(tokens_b) + 1)
    )

    attention_mask = [1] * len(input_ids)

    input_ids, mlm_labels = apply_mlm(
        input_ids=input_ids,
        vocab_size=vocab_size,
        mask_token_id=mask_token_id,
        special_token_ids={
            cls_token_id,
            sep_token_id,
            pad_token_id,
        },
    )

    return {
        "input_ids": input_ids,
        "token_type_ids": torch.tensor(
            token_type_ids,
            dtype=torch.long,
        ),
        "attention_mask": torch.tensor(
            attention_mask,
            dtype=torch.long,
        ),
        "mlm_labels": mlm_labels,
        "nsp_label": torch.tensor(
            nsp_label,
            dtype=torch.long,
        ),
    }


if __name__ == "__main__":

    from pathlib import Path

    DATA_DIR = (
        Path(__file__).resolve().parents[3]
        / "data/wikitext2/tok_wordpiece_bert_data_build/"
    )

    tokenizer = load_tokenizer(DATA_DIR / "tokenizer.json")

    documents = torch.load(
        DATA_DIR / "train_ids.pt",
        weights_only=True,
    )

    cls_token_id = tokenizer.token_to_id("[CLS]")
    sep_token_id = tokenizer.token_to_id("[SEP]")
    mask_token_id = tokenizer.token_to_id("[MASK]")
    pad_token_id = tokenizer.token_to_id("[PAD]")

    vocab_size = tokenizer.get_vocab_size()

    # ---------------------------------------------------------
    # Sample one sentence pair
    # ---------------------------------------------------------
    sampler = SentencePairSampler(
        documents=documents,
        max_seq_len=128,
    )


    print("BERT Current Sampling Schedule")
    print(sampler.i_sample)
    samples = sampler.sample(batch_size=1)
    tokens_a, tokens_b, nsp_label = samples[0]
    print("BERT Next Sampling Schedule")
    print(sampler.i_sample)

    # ---------------------------------------------------------
    # Build one complete BERT training example
    # ---------------------------------------------------------
    example = make_example(
        tokens_a=tokens_a,
        tokens_b=tokens_b,
        nsp_label=nsp_label,
        vocab_size=vocab_size,
        cls_token_id=cls_token_id,
        sep_token_id=sep_token_id,
        mask_token_id=mask_token_id,
        pad_token_id=pad_token_id,
    )

    print("[MASK] Token IDs:")
    print(mask_token_id)

    print("Input IDs:")
    print(example["input_ids"])

    print("\nToken type IDs:")
    print(example["token_type_ids"])

    print("\nAttention mask:")
    print(example["attention_mask"])

    print("\nMLM labels:")
    print(example["mlm_labels"])

    print("\nNSP label:")
    print(example["nsp_label"])

    print("\nSequence length:")
    print(len(example["input_ids"]))

    tokens = tokenizer.decode_batch(
        [example["input_ids"].tolist()]
        )[0].split()

    labels = [
        tokenizer.id_to_token(token_id)
        if token_id != -100
        else "---"
        for token_id in example["mlm_labels"].tolist()
    ]

    print("\nMLM check:")
    for i, (token, label) in enumerate(zip(tokens, labels)):
        if label != "---":
            print(f"{i:3}: {token:15} <- {label}")

        input_tokens = [
        tokenizer.id_to_token(token_id)
        for token_id in example["input_ids"].tolist()
    ]

    label_tokens = [
        tokenizer.id_to_token(token_id)
        if token_id != -100
        else "---"
        for token_id in example["mlm_labels"].tolist()
    ]

    print("\nMLM check:")
    print(f"{'Pos':>3}  {'Input':<20} {'Target':<20} {'Corruption'}")
    print("-" * 65)

    for i, (input_token, label_token) in enumerate(
        zip(input_tokens, label_tokens)
    ):
        if label_token == "---":
            continue

        if input_token == "[MASK]":
            corruption = "MASK"
        elif input_token == label_token:
            corruption = "unchanged"
        else:
            corruption = "random token"

        print(
            f"{i:3}  "
            f"{input_token:<20} "
            f"{label_token:<20} "
            f"{corruption}"
        )