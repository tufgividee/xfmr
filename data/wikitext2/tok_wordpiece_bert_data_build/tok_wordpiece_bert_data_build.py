from pathlib import Path

import spacy
import torch
from tokenizers import Tokenizer
from tokenizers.decoders import WordPiece as WordPieceDecoder
from tokenizers.models import WordPiece
from tokenizers.normalizers import BertNormalizer
from tokenizers.pre_tokenizers import BertPreTokenizer
from tokenizers.trainers import WordPieceTrainer

nlp = spacy.blank("en")
nlp.add_pipe("sentencizer")


# Setup directory
SCRIPT_DIR = Path(__file__).resolve().parent

# Raw data directory
RAW_DIR = SCRIPT_DIR.parent / "raw"


def setup_tokenizer() -> Tokenizer:
    tokenizer = Tokenizer(
        WordPiece(
            unk_token="[UNK]",
            continuing_subword_prefix="##",
        )
    )

    tokenizer.normalizer = BertNormalizer(
        lowercase=True,
        strip_accents=True,
    )
    tokenizer.pre_tokenizer = BertPreTokenizer()
    tokenizer.decoder = WordPieceDecoder(prefix="##")

    return tokenizer


def make_book_file(
        raw_file_name: str,
        book_file_name: str,
    ):
    print(f"Parsing {raw_file_name} into book...")

    documents = []
    document = []

    with open(RAW_DIR / raw_file_name, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            # Level-1 heading = new document
            if line.startswith("=") and not line.startswith("= ="):
                if document:
                    documents.append(" ".join(document))
                    document = []
                continue

            # Ignore section/subsection headings
            if line.startswith("="):
                continue

            document.append(line)

    # Save final document
    if document:
        documents.append(" ".join(document))

    def _split_into_sentences(text: str) -> list[str]:
        doc = nlp(text)
        return [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    save_path = SCRIPT_DIR / f"{book_file_name}.book"

    with open(save_path, "w", encoding="utf-8") as f:
        for document in documents:
            if len(document) <= 50:
                continue

            sentences = _split_into_sentences(document)

            for sentence in sentences:
                f.write(sentence)
                f.write("\n")

            # Blank line separates documents
            f.write("\n")

    print(
        f"Saved split -> {save_path.name} "
        f"({len(documents)} documents)"
    )


def get_training_documents(book_file_name: str):
    book_path = SCRIPT_DIR / f"{book_file_name}.book"

    with open(book_path, "r", encoding="utf-8") as f:
        for document in f.read().split("\n\n"):
            document = document.strip()

            if document:
                yield document


def make_tokenizer(book_file_name: str) -> Tokenizer:
    tokenizer = setup_tokenizer()

    trainer = WordPieceTrainer(
        special_tokens=[
            "[UNK]",
            "[PAD]",
            "[CLS]",
            "[SEP]",
            "[MASK]",
        ],
        vocab_size=30000,
        min_frequency=2,
        continuing_subword_prefix="##",
    )

    print(
        f"Training WordPiece vocabulary "
        f"using {book_file_name}.book..."
    )

    tokenizer.train_from_iterator(
        get_training_documents(book_file_name),
        trainer,
    )

    save_path = SCRIPT_DIR / "tokenizer.json"
    tokenizer.save(str(save_path))

    print(f"Saved tokenizer -> {save_path.name}")
    print(f"Vocabulary size: {tokenizer.get_vocab_size()}")

    return tokenizer


def make_data_file(
    book_file_name: str,
    data_file_name: str,
    tokenizer: Tokenizer,
):
    print(f"Tokenizing {book_file_name}.book...")

    book_path = SCRIPT_DIR / f"{book_file_name}.book"

    with open(book_path, "r", encoding="utf-8") as f:
        documents = [
            document.strip()
            for document in f.read().split("\n\n")
            if document.strip()
        ]

    tokenized_documents = []
    total_tokens = 0

    for document in documents:
        sentences = document.splitlines()

        tokenized_sentences = [
            tokenizer.encode(sentence).ids
            for sentence in sentences
            if sentence.strip()
        ]

        tokenized_documents.append(tokenized_sentences)

        total_tokens += sum(
            len(sentence)
            for sentence in tokenized_sentences
        )

    save_path = SCRIPT_DIR / f"{data_file_name}.pt"
    torch.save(tokenized_documents, save_path)

    print(
        f"Saved data -> {save_path.name} "
        f"({len(tokenized_documents)} documents, "
        f"{total_tokens:,} tokens)"
    )

def inspect_structure(data, depth=0, max_depth=3):
    indent = "  " * depth

    if depth > max_depth:
        print(f"{indent}...")
        return

    if isinstance(data, list):
        print(f"{indent}list[{len(data)}]")

        if data:
            inspect_structure(data[0], depth + 1, max_depth)

    else:
        print(f"{indent}{type(data).__name__}")


if __name__ == "__main__":
    print(
        "Making tokenizer and datasets "
        "for WikiText-2 (WordPiece)..."
    )

    # 0. Parse raw WikiText into document chunks.
    make_book_file(
        "wiki.train.raw",
        "wiki.train",
    )

    make_book_file(
        "wiki.valid.raw",
        "wiki.valid",
    )

    make_book_file(
        "wiki.test.raw",
        "wiki.test",
    )

    # 1. Train ONE shared tokenizer using training data only.
    tokenizer = make_tokenizer("wiki.train")

    # 2. Encode all splits using the same tokenizer.
    splits = {
        "train": "wiki.train",
        "val": "wiki.valid",
        "test": "wiki.test",
    }

    for split, book_file in splits.items():
        make_data_file(
            book_file,
            f"{split}_ids",
            tokenizer,
        )

    print("WordPiece data preparation complete!")
    data = torch.load("train_ids.pt", weights_only=True)
    inspect_structure(data)



