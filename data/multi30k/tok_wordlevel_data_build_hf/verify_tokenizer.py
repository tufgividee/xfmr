from detokenizer import detokenize

from xfmr.xfmr2017.data import load_tokenizer

# Gain insight into the tokenizer by encoding and decoding some sample text
txt_en = "A little girl climbing into a wooden playhouse."
txt_de = "Ein kleines Mädchen klettert in ein Spielhaus aus Holz."

tokenizer_en = load_tokenizer("tokenizer_en.json")
tokenizer_de = load_tokenizer("tokenizer_de.json")

enc_en = tokenizer_en.encode(txt_en)
dec_en = tokenizer_en.decode(enc_en.ids)

print(txt_en)
print(enc_en.ids)
print(enc_en.tokens)


enc_de = tokenizer_de.encode(txt_de)
dec_de = tokenizer_de.decode(enc_de.ids)
print(txt_de)
print(enc_de.ids)
print(enc_de.tokens)
print(detokenize(tokenizer_de.decode(enc_de.ids)))


# Test the tokenizer with some words to see how it handles them
test_en_words = [
    "playhouse",
    "playing",
    "beautiful",
    "international",
]

test_de_words = [
    "Spielhaus",
    "Mädchen",
    "klettert",
    "Holz",
]

for word in test_en_words:
    enc = tokenizer_en.encode(word)
    print(f"{word:20} -> {enc.tokens}")

for word in test_de_words:
    enc = tokenizer_de.encode(word)
    print(f"{word:20} -> {enc.tokens}")


### 

# Continue onto learning about tokenizer, seeking a way to handle added spaces

print(f"tokenizer_en: {tokenizer_en}")
print(f"tokenizer_de: {tokenizer_de}")

print(f"decoder_en: {tokenizer_en.decoder}")
print(f"decoder_de: {tokenizer_de.decoder}")
print(f"pre_tokenizer_en: {tokenizer_en.pre_tokenizer}")
print(f"pre_tokenizer_de: {tokenizer_de.pre_tokenizer}")
print(f"post_processor_en: {tokenizer_en.post_processor}")
print(f"post_processor_de: {tokenizer_de.post_processor}")

print(f"enc_en.tokens: {enc_en.tokens}")
print(f"tokenizer_en.decode(enc_en.ids): {tokenizer_en.decode(enc_en.ids)}")
print(f"detokenize(tokenizer_en.decode(enc_en.ids)): {detokenize(tokenizer_en.decode(enc_en.ids))}")
print(f"enc_de.tokens: {enc_de.tokens}")
print(f"tokenizer_de.decode(enc_de.ids): {tokenizer_de.decode(enc_de.ids)}")
print(f"detokenize(tokenizer_de.decode(enc_de.ids)): {detokenize(tokenizer_de.decode(enc_de.ids))}")

print(f"(tokenizer_de.encode('spielhaus')).ids: {(tokenizer_de.encode('spielhaus')).ids}")