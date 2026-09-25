import random

"""
Reference: token-level random A/B sampler.

Currently unused. The BERT pipeline uses sentence-level sampling
instead. Kept here as a reference for alternative sampling strategies.

coverage guidance:
tot_tok / ( (128-3 + 2x min_seg_len) / 2 )

"""

def pick_document_candidate(documents, min_doc_len):
    candidates = [
        doc for doc in documents
        if len(doc) >= min_doc_len
    ]

    if not candidates:
        raise ValueError(
            "No document is long enough for the requested sample."
        )

    return random.choice(candidates)

def make_nsp_pair(
    documents: list[list[int]],
    max_seq_len: int = 128, # 512
    min_seg_len: int = 2, # segment a or b length
) -> tuple[list[int], list[int], int]:
\
    max_pair_tokens = max_seq_len - 3  # [CLS], [SEP], [SEP]

    # Choose total A+B length.
    total_len = random.randint(
        min_seg_len * 2,
        max_pair_tokens,
    )

    # Randomly divide total length between A and B.
    seg_a_len = random.randint(
        min_seg_len,
        total_len - min_seg_len,
    )
    seg_b_len = total_len - seg_a_len

    # 50% IsNext / 50% NotNext.
    is_next = random.random() < 0.5

    if is_next:
        # A and B are contiguous parts of the same document.
        document = pick_document_candidate(documents, total_len)

        start = random.randint(
            0,
            len(document) - total_len,
        )

        chunk = document[start:start + total_len]

        tokens_a = chunk[:seg_a_len]
        tokens_b = chunk[seg_a_len:]

    else:
        # A and B come from different documents.
        document_a = pick_document_candidate(documents, seg_a_len)
        document_b = pick_document_candidate(documents, seg_b_len)

        start_a = random.randint(
            0,
            len(document_a) - seg_a_len,
        )

        start_b = random.randint(
            0,
            len(document_b) - seg_b_len,
        )

        tokens_a = document_a[start_a:start_a + seg_a_len]
        tokens_b = document_b[start_b:start_b + seg_b_len]

    nsp_label = 0 if is_next else 1

    return tokens_a, tokens_b, nsp_label