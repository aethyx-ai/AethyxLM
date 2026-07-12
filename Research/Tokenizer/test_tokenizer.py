from Research.Tokenizer.tokenizer import AethyxTokenizer


training_text = """
Hello world.
Hello Ishaan!
Welcome to Aethyx Labs.
This is the first tokenizer.
"""


tokenizer = AethyxTokenizer()

tokenizer.build_vocab(training_text)

tokenizer.save("vocab.json")

print("=" * 50)

print("Vocabulary")

for word, idx in tokenizer.word_to_id.items():

    print(f"{idx:>3} -> {word}")

print("=" * 50)

sentence = "Hello bro! Welcome."

encoded = tokenizer.encode(sentence)

print("\nOriginal")

print(sentence)

print("\nEncoded")

print(encoded)

decoded = tokenizer.decode(encoded)

print("\nDecoded")

print(decoded)

tokenizer2 = AethyxTokenizer()

tokenizer2.load("vocab.json")

print("\nLoaded Vocabulary Size")

print(len(tokenizer2.word_to_id))