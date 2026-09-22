import re


def detokenize(text: str) -> str:
    # Remove spaces before punctuation.
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)

    # Remove spaces after opening brackets.
    text = re.sub(r"([\(\[\{])\s+", r"\1", text)

    # Remove spaces before closing brackets.
    text = re.sub(r"\s+([\)\]\}])", r"\1", text)

    # Join common English contractions.
    text = re.sub(r"\s+('s|'re|'ve|'ll|'d|'m|'t)\b", r"\1", text)

    # Join "n't".
    text = re.sub(r"\s+n['’]t\b", "n't", text)

    # Remove spaces immediately inside double quotes.
    text = re.sub(r'"\s+', '"', text)
    text = re.sub(r'\s+"', '"', text)

    return text