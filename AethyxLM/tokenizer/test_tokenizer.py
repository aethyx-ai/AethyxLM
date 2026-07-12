"""
AethyxLM Tokenizer Test
"""

from .tokenizer import AethyxTokenizer


def main():

    tokenizer = AethyxTokenizer()

    text = "Hello Aethyx! Artificial Intelligence is awesome."

    print("=" * 60)

    print("Original:")
    print(text)

    print("\nTokens:")
    print(tokenizer.tokenize(text))

    print("\nToken IDs:")
    ids = tokenizer.encode(text)
    print(ids)

    print("\nDecoded:")
    print(tokenizer.decode(ids))

    print("\nVocabulary Size:")
    print(tokenizer.vocab_size)

    print("=" * 60)


if __name__ == "__main__":
    main()