# AethyxLM

> Building the next generation of frontier AI systems from India.

AethyxLM is a decoder-only Large Language Model built completely from scratch in **PyTorch** by **Aethyx AI**.

Unlike projects that rely heavily on existing implementations, AethyxLM is being engineered from the ground up to understand, improve, and eventually redefine modern language model architectures.

---

## 🎯 Vision

Our long-term mission is to build globally competitive **frontier AI models** originating from India.

We're not building another AI wrapper or chatbot—we're building the foundation itself.

Beyond scaling model capability, we're researching new approaches to make large language models significantly more efficient, particularly in how they process and utilize context.

---

## 🚀 Current Progress

### ✅ Core Architecture

- Custom GPT-style Decoder Architecture
- Multi-Head Causal Self-Attention
- Custom LayerNorm
- Feed Forward Network (GELU)
- Learned Token & Positional Embeddings
- Pre-LayerNorm Transformer Blocks
- Language Modeling Head

### ✅ Training Infrastructure

- Cross-Entropy Language Modeling Loss
- AdamW Optimizer
- Cosine Learning Rate Scheduler with Warmup
- Mixed Precision (AMP)
- Gradient Clipping
- Gradient Accumulation
- Checkpoint Save & Resume
- Modular Training Pipeline

### ✅ Inference

- Local Chat Interface
- Token Sampling
- Temperature Control
- Top-k Sampling
- Grouped-query attention and KV-cached decoding
- Scaled-dot-product attention with optimized GPU kernel dispatch

### Experimental Context Interface

- Opt-in typed context adapter with a fixed latent budget
- Selected-layer gated cross-attention
- Representation-agnostic boundary for future local context compilers
- Compression metrics coupled to downstream accuracy retention
- Query-aware local retrieval with exact-source references
- Optional sliding-window/global-layer long-context experiments

This interface is research infrastructure, not a validated context-compression
capability. The target compression ratio remains an experimental hypothesis.

---

## 🛣️ Roadmap

### Phase 1 — Foundation
- [x] Project Setup
- [x] GPT Decoder Architecture
- [x] Training Infrastructure
- [x] Inference Pipeline
- [ ] Complete TinyStories Pretraining

### Phase 2 — Scaling
- [ ] Larger Models
- [ ] Distributed Training
- [ ] Advanced Evaluation
- [ ] Long Context Support
- [ ] Efficient Inference

### Phase 3 — Research
- [ ] Novel Context Compression
- [ ] Persistent AI Memory
- [ ] Advanced Reasoning
- [ ] Multimodal Capabilities
- [ ] Frontier-Scale Models

---

## 💡 Research Direction

One of AethyxLM's primary research directions is improving **context efficiency**.

Current transformer models repeatedly process large amounts of text—system prompts, conversation history, retrieved documents, and tool definitions—resulting in significant computational overhead.

We're exploring methods to represent large contexts in more compact forms with the goal of reducing effective context costs while preserving useful information.

---

## 🛠️ Tech Stack

- Python
- PyTorch
- CUDA
- Git & GitHub

---

## 📂 Project Structure

```text
AethyxLM/
├── model/
├── training/
├── data/
├── tokenizer/
├── utils/
├── checkpoints/
├── train.py
└── chat.py
```

---

## 📈 Project Status

AethyxLM is currently in active development.

The model architecture, training infrastructure, and inference pipeline have been built and validated. Our current focus is scaling training, advancing research, and developing more efficient transformer architectures.

---

## 🤝 Contributing

Contributions, discussions, and research collaborations are always welcome.

If you're interested in working on efficient transformers, training systems, or frontier AI research, feel free to open an issue or start a discussion.

---

## 📄 License

This project is licensed under the MIT License.

---

<p align="center">
Building the future of AI — one layer at a time.
</p>
