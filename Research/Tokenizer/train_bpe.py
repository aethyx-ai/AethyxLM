from Research.Tokenizer.bpe_trainer import BPETrainer


text = """
hello world

hello world

hello ishaan

hello aethyx

artificial intelligence

machine learning

deep learning

transformers

language models

language model

tokenizer tokenizer tokenizer
"""


trainer = BPETrainer()

trainer.train(
    text,
    num_merges=50
)

trainer.save_merges(
    "merges.json"
)

trainer.save_vocab(
    "vocab.json"
)

print("\nTraining Finished.")