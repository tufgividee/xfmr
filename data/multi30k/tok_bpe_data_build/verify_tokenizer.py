from xfmr.xfmr2017.data import load_tokenizer

# Gain insight into the tokenizer by encoding and decoding some sample text
txt_en = "A little girl climbing into a wooden playhouse."
txt_de = "Ein kleines Mädchen klettert in ein Spielhaus aus Holz."

tokenizer = load_tokenizer("tokenizer.json")

enc_en = tokenizer.encode(txt_en)
dec_en = tokenizer.decode(enc_en.ids)

print(txt_en)
print(enc_en.ids)
print(enc_en.tokens)
print()


enc_de = tokenizer.encode(txt_de)
dec_de = tokenizer.decode(enc_de.ids)
print(txt_de)
print(enc_de.ids)
print(enc_de.tokens)
print(tokenizer.decode(enc_de.ids))


# Test the tokenizer with some words to see how it handles them
test_words = [
    "playhouse",
    "playing",
    "beautiful",
    "international",
    "Spielhaus",
    "Mädchen",
    "klettert",
    "Holz",
]

for word in test_words:
    enc = tokenizer.encode(word)
    print(f"{word:20} -> {enc.tokens}")


# Obtain BLEU scores for different scenarios with tokenizer in the loop
import sacrebleu

# 1. Raw vs raw
score1 = sacrebleu.corpus_bleu(
    [txt_en],
    [[txt_en]],
)

# 2. Tokenizer-decoded vs raw
score2 = sacrebleu.corpus_bleu(
    [dec_en],
    [[txt_en]],
)

# 3. Tokenizer-decoded vs tokenizer-decoded
score3 = sacrebleu.corpus_bleu(
    [dec_en],
    [[dec_en]],
)

# 4. Tokenizer-decoded vs lower(raw)
score4 = sacrebleu.corpus_bleu(
    [dec_en],
    [[txt_en.lower()]],
)



print(f"Raw vs raw:       {score1.score:.2f}")
print(f"Decoded vs raw:   {score2.score:.2f}")
print(f"Decoded vs decoded: {score3.score:.2f}")
print(f"Decoded vs lower(raw):   {score4.score:.2f}")

print()
print("Raw:    ", repr(txt_en))
print("Decoded:", repr(dec_en))
print('------------------------------------------------')

# Manually examination of BLEU scores
def noodle_bleu(reference, hypothesis):
    score = sacrebleu.corpus_bleu(
        [hypothesis],
        [[reference]],
    )

    print("BLEU:", score.score)
    print("Reference :", repr(reference))
    print("Hypothesis:", repr(hypothesis))


noodle_bleu(
    reference="A little girl climbing into a wooden playhouse.",
    hypothesis="a little girl climbing into a wooden playhouse.",
)

noodle_bleu(
    reference="A little girl climbing into a wooden playhouse.",
    hypothesis="a little girl climbing into a wooden play house.",
)

### 

# Continue onto learning about tokenizer, seeking a way to handle added spaces

print(tokenizer)

print(tokenizer.decoder)
print(tokenizer.pre_tokenizer)
print(tokenizer.post_processor)

print(tokenizer.decode(enc_en.ids))
print(tokenizer.decode(enc_en.ids, skip_special_tokens=True))