# AethyxLM Project Memory

## Project Overview
Building a GPT-style decoder-only LLM from scratch in PyTorch to understand every component. No reliance on HuggingFace internals.

## Philosophy & Rules
- Build everything ourselves whenever reasonably possible
- Learn every mathematical component before moving on
- Keep code clean, modular, production-quality
- Never copy HuggingFace internals blindly
- Every module gets its own test file before moving on
- Never continue until all tests pass
- Explain concepts before implementation
- Think like OpenAI/Anthropic engineers, not tutorial followers

## Model Configuration (config.py)
```python
VOCAB_SIZE = 32000
CONTEXT_LENGTH = 128
EMBED_DIM = 256
NUM_HEADS = 8
HEAD_DIM = 32  # EMBED_DIM // NUM_HEADS
NUM_LAYERS = 8
FFN_DIM = EMBED_DIM * 4  # 1024
DROPOUT = 0.1
USE_BIAS = True
LAYER_NORM_EPS = 1e-5
```

## Architecture Summary
- Decoder-only GPT
- Learned positional embeddings
- Learned token embeddings
- Multi-head causal self-attention
- Pre-LayerNorm
- Residual connections
- Feed Forward (4x hidden expansion, GELU)
- Final LayerNorm
- Linear language modeling head (tied weights)

## Implemented Modules (model/)
| File | Component | Test Status |
|------|-----------|-------------|
| `embedding.py` | TokenEmbedding | ✅ |
| `positional_embedding.py` | PositionalEmbedding | ✅ |
| `attention.py` | MultiHeadSelfAttention | ✅ |
| `feed_forward.py` | FeedForward (Linear-GELU-Linear-Dropout) | ✅ |
| `layer_norm.py` | LayerNorm (custom, no nn.LayerNorm) | ✅ |
| `transformer_block.py` | TransformerBlock (Pre-Norm) | ✅ |
| `gpt.py` | Complete GPT Model | ✅ |

**Test Commands:**
```bash
python -m model.test_embedding
python -m model.test_positional_embedding
python -m model.test_attention
python -m model.test_feed_forward
python -m model.test_layer_norm
python -m model.test_transformer_block
python -m model.test_gpt
```

## GPT Forward Pass
```
input_ids (batch, seq_len)
    ↓ TokenEmbedding + PositionalEmbedding
    ↓ Dropout
    ↓ 8 × TransformerBlock (Pre-Norm: LN → Attention → Residual → LN → FFN → Residual)
    ↓ Final LayerNorm
    ↓ LM Head (tied weights)
logits (batch, seq_len, vocab_size)
```

**Output Shape:** `(8, 128, 32000)` — No NaN, No Inf

## Training Infrastructure (training/) ✅ COMPLETE
| File | Component | Test Status |
|------|-----------|-------------|
| `loss.py` | LanguageModelLoss (CrossEntropy) | ✅ |
| `optimizer.py` | AdamW with weight decay separation | ✅ |
| `scheduler.py` | Warmup + Cosine Decay LR scheduler | ✅ |
| `trainer.py` | Training loop (grad clip, AMP, checkpoints) | ✅ |
| `test_training.py` | All training component tests | ✅ |

**Test Command:**
```bash
python -m training.test_training
```

## Data Pipeline (data/) ✅ COMPLETE
| File | Component | Test Status |
|------|-----------|-------------|
| `dataset.py` | AethyxDataset (tokenized sliding window) | ✅ |
| `dataloader.py` | create_dataloader() | ✅ |
| `config.py` | Data configuration | ✅ |

**Test Commands:**
```bash
python -m data.test_dataset
python -m data.test_dataloader
```

## Tokenizer (tokenizer/) ✅ COMPLETE
- BPE tokenizer using HuggingFace tokenizers (production quality)
- Config: VOCAB_SIZE=32000, special tokens: PAD, UNK, BOS, EOS
- `train_tokenizer.py` trains on corpus.txt
- Wrapper class `AethyxTokenizer` with encode/decode/tokenize

## Current Stage
**Complete GPT architecture + training infrastructure built and tested.** Ready for first real training run on TinyStories or similar dataset.

## Next Milestones
1. **Download/prepare TinyStories dataset** (or similar)
2. **Train tokenizer on full corpus** 
3. **First training run on real data** (GPU recommended - CPU is slow)
4. **Implement text generation / inference** (`inference/generate.py`)
5. **Evaluation metrics** (perplexity, etc.)
6. **Scaling to larger models** (125M, 350M, 1B+)

## Directory Structure
```
AethyxLM/
├── model/
│   ├── config.py
│   ├── embedding.py
│   ├── positional_embedding.py
│   ├── attention.py
│   ├── feed_forward.py
│   ├── layer_norm.py
│   ├── transformer_block.py
│   ├── gpt.py
│   └── test_*.py
├── training/
│   ├── loss.py
│   ├── optimizer.py
│   ├── scheduler.py
│   ├── trainer.py
│   └── test_training.py
├── data/
│   ├── dataset.py
│   ├── dataloader.py
│   ├── config.py
│   └── test_*.py
├── tokenizer/
│   ├── tokenizer.py
│   ├── train_tokenizer.py
│   └── config.py
├── train.py
├── chat.py
└── checkpoints/
```

## Tokenizer Status
- Production tokenizer in `tokenizer/` with config (VOCAB_SIZE=32000)
- `train_tokenizer.py` for training on corpus
- Small test corpus in `tokenizer/data/corpus.txt` and `dataset/corpus.txt`

## Long-Term Roadmap
Tiny GPT → Tokenizer → DataLoader → Training Loop → Train on TinyStories → Inference → Optimization → Scale to larger models → Train custom ~1B parameter model.