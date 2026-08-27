# AethyxLM

AethyxLM is an Indian language-model research platform focused on building efficient, capable transformer systems.

## Vision

AethyxLM aims to become a future Indian frontier-model platform. One long-term research priority is improving how language models handle large amounts of context while preserving useful information and response quality. The underlying research and implementation remain private and experimental.

## Current Model

The current AethyxLM implementation is a decoder-only transformer and a stable foundation for continued architecture research. Its v3 research configuration has approximately **137.6 million parameters** and includes:

- A 48,000-token custom ByteLevel BPE tokenizer
- Grouped-query attention with fused QKV projections
- Scaled dot-product attention
- Rotary positional embeddings
- RMSNorm and SwiGLU feed-forward layers
- Query/key normalization
- Gradient checkpointing and mixed-precision support
- Sliding-window and periodic global-attention support
- KV-cached generation with configurable sampling controls

## Project Status

AethyxLM is under active private development by AETHYX Labs. Architecture, interfaces, and research results may change as the project evolves.

## License

This project is proprietary. All rights are reserved; commercial use, redistribution, copying, or modification is not permitted without prior written authorization from AETHYX Labs. See [LICENSE](LICENSE).
