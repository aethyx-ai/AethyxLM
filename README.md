# AethyxLM

AethyxLM is an Indian language-model research platform focused on building efficient, capable transformer systems and rethinking how models consume context.

## Vision

Modern language-model applications repeatedly transmit large volumes of prompts, histories, tool definitions, documents, and structured state as ordinary text tokens. AethyxLM's primary research direction is a local **context compiler** that transforms this information into a denser machine-readable representation before it reaches the model.

The representation is not assumed to be visual. Research candidates include structured graphs, spatial or visual context pages, learned latent memory, and hybrid text representations. The objective is to preserve useful information and task quality while reducing token usage, bandwidth, memory pressure, and inference cost. Compression targets remain hypotheses until validated experimentally.

## Current Model

The current AethyxLM implementation is a compact decoder-only transformer and a stable foundation for continued architecture research. Its production research configuration has approximately **31.2 million parameters** and includes:

- A 32,000-token custom ByteLevel BPE tokenizer
- Grouped-query attention with fused QKV projections
- Scaled dot-product attention
- Rotary positional embeddings
- RMSNorm and SwiGLU feed-forward layers
- Query/key normalization
- Gradient checkpointing and mixed-precision support
- Sliding-window and periodic global-attention support
- Experimental interfaces for future compressed-context adapters

The compressed-context layer is still an active research direction, not a claimed production result. The language-model pipeline remains modular so that multimodal, graph-based, retrieval, latent-memory, and persistent-agent-state approaches can be evaluated without destabilizing the core model.

## Project Status

AethyxLM is under active private development by AETHYX Labs. Architecture, interfaces, and research results may change as the project evolves.

## License

This project is proprietary. All rights are reserved; commercial use, redistribution, copying, or modification is not permitted without prior written authorization from AETHYX Labs. See [LICENSE](LICENSE).
