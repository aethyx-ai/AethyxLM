# Graph Report - D:\CODING\AETHYXLabs  (2026-08-12)

## Corpus Check
- Large corpus: 238 files · ~1,413,981 words. Semantic extraction will be expensive (many Claude tokens). Consider running on a subfolder.

## Summary
- 932 nodes · 1814 edges · 50 communities (42 shown, 8 thin omitted)
- Extraction: 96% EXTRACTED · 4% INFERRED · 0% AMBIGUOUS · INFERRED: 65 edges (avg confidence: 0.62)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Context Compilation Core
- Dataset Streaming Preparation
- Training Data Pipeline
- Evaluation and Generation
- Transformer Attention Blocks
- Context Adapter Modules
- Architecture and Research Vision
- Tokenizer Model Benchmarks
- GPU Training Diagnostics
- Trainer Checkpoint Lifecycle
- Interactive Chat Interface
- Rotary Position Encoding
- Instruction Fine Tuning
- GPT Model Compatibility
- Context Research Contracts
- Core Neural Layers
- Training Scheduler Setup
- Loss and Optimizer Debugging
- Tokenizer V2 Validation
- Training Integration Tests
- Legacy Position Components
- Milestone Checkpoint Management
- Data Loading Tests
- Persistent Context Memory
- Experiment Event Tracking
- Legacy BPE Training
- Token Embedding Layer
- Layer Normalization
- Multilingual Corpus Preparation
- Legacy Tokenizer Implementation
- Scaling Law Analysis
- CUDA Scaling Pilots
- Weight Initialization
- Browser Context Compiler
- Tokenizer Text Utilities
- Project Dependencies
- Kaggle Notebook Builder
- Training Corpora
- Tensor Forward Contracts
- SDPA Backend Probes
- Experiment Summaries
- Checkpoint Suite Tests
- Training Configuration
- Evaluation Package

## God Nodes (most connected - your core abstractions)
1. `GPT` - 88 edges
2. `AethyxTokenizer` - 61 edges
3. `Trainer` - 37 edges
4. `AethyxDataset` - 27 edges
5. `FineWebPreparer` - 27 edges
6. `LanguageModelLoss` - 25 edges
7. `create_optimizer()` - 25 edges
8. `BinaryTokenWriter` - 21 edges
9. `create_dataloader()` - 19 edges
10. `LocalContextCompiler` - 18 edges

## Surprising Connections (you probably didn't know these)
- `Production 31.2M-Parameter Model` --semantically_similar_to--> `Efficient Transformer Baseline`  [INFERRED] [semantically similar]
  README.md → AethyxLM/ARCHITECTURE.md
- `Context Compiler Vision` --semantically_similar_to--> `Local Context Compilation Research`  [INFERRED] [semantically similar]
  README.md → AethyxLM/docs/CONTEXT_COMPRESSION_RESEARCH.md
- `32K BPE Tokenizer Plan` --semantically_similar_to--> `Tokenizer v2 32K ByteLevel BPE`  [INFERRED] [semantically similar]
  MEMORY.md → AethyxLM/ARCHITECTURE.md
- `Understanding-First Engineering Principles` --semantically_similar_to--> `Test-First Module Development`  [INFERRED] [semantically similar]
  AethyxLM/docs/AethyxLM_SPEC.md → MEMORY.md
- `Legacy GPT-Style Architecture` --conceptually_related_to--> `AethyxLM Modern Transformer Architecture`  [INFERRED]
  MEMORY.md → AethyxLM/ARCHITECTURE.md

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Modern Transformer Component Family** — aethyxlm_architecture_efficient_baseline, aethyxlm_architecture_grouped_query_attention, aethyxlm_architecture_rope_aware_kv_cache, aethyxlm_architecture_sliding_window_attention [EXTRACTED 1.00]
- **Local Context Compilation Pipeline** — aethyxlm_docs_context_compression_research_verbatim_risk_classification, aethyxlm_docs_context_compression_research_deterministic_graph_extraction, aethyxlm_docs_context_compression_research_query_scoped_source_selection, aethyxlm_docs_context_compression_research_retention_cost_benchmark [EXTRACTED 1.00]
- **AethyxLM Research System Family** — aethyxlm_docs_research_systems_stable_pretraining, aethyxlm_docs_research_systems_checkpoint_evaluation_suite, aethyxlm_docs_research_systems_local_experiment_tracking, aethyxlm_docs_research_systems_instruction_fine_tuning, aethyxlm_docs_research_systems_generation_engine, aethyxlm_docs_research_systems_context_lab [EXTRACTED 1.00]

## Communities (50 total, 8 thin omitted)

### Community 0 - "Context Compilation Core"
Cohesion: 0.08
Nodes (44): benchmark_compiler(), BenchmarkCase, Representation benchmark that scores reduction only with task retention., CompilerPolicy, LocalContextCompiler, Local hybrid compiler with graph retrieval and conservative visual gating., Compile context locally while preserving exact-risk and provenance data., DeterministicContextGraph (+36 more)

### Community 1 - "Dataset Streaming Preparation"
Cohesion: 0.06
Nodes (35): completed_output(), ensure_token_sidecars(), prepare_bundle(), Path, Prepare a manifest-defined collection without retaining raw source text., atomic_write_json(), BinaryTokenWriter, FineWebPreparer (+27 more)

### Community 2 - "Training Data Pipeline"
Cohesion: 0.05
Nodes (44): AethyxLM Dataset Configuration, AethyxDataset, MixedAethyxDataset, Dataset, Path, AethyxLM Dataset Converts raw text into training samples. Supports .txt, .csv,…, Initialize worker with unique seed., Memory-efficient dataset that stores tokenised data as a memory-mapped numpy… (+36 more)

### Community 3 - "Evaluation and Generation"
Cohesion: 0.07
Nodes (44): evaluate_checkpoint(), evaluate_generation(), evaluate_registry(), load_checkpoint_once(), Path, Checkpoint-wide multilingual, generation, and long-context evaluation., _repetition_metrics(), context_compression_metrics() (+36 more)

### Community 4 - "Transformer Attention Blocks"
Cohesion: 0.06
Nodes (41): MultiHeadSelfAttention, Tensor, Expand grouped K/V heads without allocating when possible., Causal self-attention with optional grouped key/value heads., build_feedforward(), FeedForward, Module, Tensor (+33 more)

### Community 5 - "Context Adapter Modules"
Cohesion: 0.06
Nodes (36): ContextCrossAttention, LatentContextAdapter, Tensor, Opt-in latent interface for compiled or structured context. This module…, Inject compiled-context latents into selected decoder layers., Resample variable-length compiler features into fixed context latents., Print architecture configuration at initialization., build_normalization() (+28 more)

### Community 6 - "Architecture and Research Vision"
Cohesion: 0.05
Nodes (45): ContextMemoryBank, Efficient Transformer Baseline, Grouped-Query Attention, LatentContextAdapter, AethyxLM Modern Transformer Architecture, RMSNorm, RoPE-Aware KV-Cached Decoding, RoPE (+37 more)

### Community 7 - "Tokenizer Model Benchmarks"
Cohesion: 0.06
Nodes (28): benchmark_model(), load_model_config(), main(), Benchmark classic and modern AethyxLM execution paths., synchronize(), timed(), Tensor, Forward pass. Args: input_ids: Shape -> (batch_size, context_length) Returns:… (+20 more)

### Community 8 - "GPU Training Diagnostics"
Cohesion: 0.09
Nodes (36): AdamW, check_amp_active(), check_backward_oom(), check_batch_size_and_seq_len(), check_checkpoints(), check_computation_graph(), check_cuda_cache(), check_dataloader_config() (+28 more)

### Community 9 - "Trainer Checkpoint Lifecycle"
Cohesion: 0.10
Nodes (20): checkpoint_step(), make_checkpoint_trainer(), Path, StateStub, test_checkpoint_aliases_and_numbered_rotation(), test_training_loop_saves_completed_step_numbers(), test_training_steps_count_optimizer_updates_with_accumulation(), TrainingBatches (+12 more)

### Community 10 - "Interactive Chat Interface"
Cohesion: 0.13
Nodes (29): checkpoint_step(), discover_checkpoints(), generate(), interactive_chat(), load_model_and_tokenizer(), main(), newest_checkpoint(), parse_args() (+21 more)

### Community 11 - "Rotary Position Encoding"
Cohesion: 0.09
Nodes (25): apply_rotary_pos_emb(), build_rope(), Tensor, Rotary Positional Embeddings (RoPE) for AethyxLM. Reference: "RoFormer:…, Apply rotary positional embeddings to input tensor. Args: x: Input tensor of…, Rotate half the hidden dimensions. Args: x: Tensor of shape (..., 2*H).…, Factory function to build RoPE module. Args: head_dim: Dimension of each…, Rotary Positional Embeddings (RoPE). Computes and caches sin/cos values for… (+17 more)

### Community 12 - "Instruction Fine Tuning"
Cohesion: 0.13
Nodes (21): iter_records(), main(), normalize_record(), Path, quality_reason(), Normalize, validate, deduplicate, and split instruction conversations., stable_key(), write_jsonl() (+13 more)

### Community 13 - "GPT Model Compatibility"
Cohesion: 0.12
Nodes (16): GPT, Complete GPT model for AethyxLM., Load current and legacy checkpoints without persisting causal masks., Decoder-only GPT Language Model., Reconstruct legacy model configuration from authoritative tensors., Load model from checkpoint with automatic architecture detection. Args:…, gen(), no_grad (+8 more)

### Community 14 - "Context Research Contracts"
Cohesion: 0.14
Nodes (15): ContextPreservationObjective, PreservationLosses, Training objectives for information-preserving context latents., Reconstruct source features and align their global semantics., CompiledContextBatch, ContextType, Tensor, Canonical tensor contract for local context-compiler output. (+7 more)

### Community 15 - "Core Neural Layers"
Cohesion: 0.17
Nodes (11): Efficient causal self-attention for AethyxLM. Supports multi-head and grouped-…, Model configuration for AethyxLM., FeedForward, Feed Forward Network (MLP) for AethyxLM. This module is applied independently…, Position-wise Feed Forward Network. Input Shape: (batch_size, sequence_length,…, Linear, Common neural network layers used throughout AethyxLM. This module provides…, Standard Linear layer. This currently behaves exactly like PyTorch's nn.Linear,… (+3 more)

### Community 16 - "Training Scheduler Setup"
Cohesion: 0.15
Nodes (14): get_constant_schedule_with_warmup(), get_cosine_schedule_with_warmup(), Learning Rate Scheduler with Warmup and Cosine Decay., Learning rate schedule with linear warmup and cosine decay. Args: optimizer:…, Constant LR with linear warmup. Args: optimizer: Optimizer to schedule…, Test learning rate scheduler., test_scheduler(), create_grad_scaler() (+6 more)

### Community 17 - "Loss and Optimizer Debugging"
Cohesion: 0.18
Nodes (10): main(), print_mem(), main(), print_mem(), LanguageModelLoss, Tensor, Cross-Entropy Loss for Language Modeling., Cross-entropy loss for next-token prediction. Handles label shifting… (+2 more)

### Community 18 - "Tokenizer V2 Validation"
Cohesion: 0.17
Nodes (12): evaluate(), load_documents(), main(), Namespace, Path, Measure tokenizer fertility and reversible normalization by language sample., test_token_cache_rejects_a_different_tokenizer(), test_v2_tokenizer_preserves_case_accents_and_structural_tokens() (+4 more)

### Community 19 - "Training Integration Tests"
Cohesion: 0.15
Nodes (14): test_sft_dataset_runs_one_assistant_masked_optimizer_step(), Tests for training components., Test Trainer initialization and single step., Test checkpoint saving and loading., Test a complete mini training run., Test LanguageModelLoss., Test AdamW optimizer creation., test_checkpoint_save_load() (+6 more)

### Community 20 - "Legacy Position Components"
Cohesion: 0.19
Nodes (9): PositionalEmbedding, Tensor, Learnable Positional Embedding Layer for AethyxLM., Learns a unique embedding vector for each position in the sequence., Args: token_embeddings: Shape -> (batch_size, sequence_length, embed_dim)…, main(), Test script for the Multi-Head Self-Attention module., main() (+1 more)

### Community 21 - "Milestone Checkpoint Management"
Cohesion: 0.24
Nodes (11): main(), Archive or inspect non-rotating training milestones., test_archive_milestone_is_non_rotating_and_manifested(), test_existing_milestone_requires_identical_content(), archive_milestone(), Any, Path, Immutable milestone checkpoint archival and manifest management. (+3 more)

### Community 22 - "Data Loading Tests"
Cohesion: 0.26
Nodes (8): create_dataloader(), AethyxLM DataLoader Creates PyTorch DataLoaders for training., main(), AethyxLM DataLoader Test, main(), Test script for the Token Embedding layer., main(), Test the complete GPT model.

### Community 23 - "Persistent Context Memory"
Cohesion: 0.26
Nodes (8): ContextMemoryBank, Tensor, Query-aware retrieval over locally compiled context features., Immutable batched context store with deterministic cosine retrieval. Source IDs…, RetrievalResult, test_entirely_empty_memory_rows_are_rejected(), test_masked_items_are_never_returned_as_valid(), test_query_aware_retrieval_preserves_types_and_source_references()

### Community 24 - "Experiment Event Tracking"
Cohesion: 0.24
Nodes (7): test_jsonl_tracker_writes_run_and_metric_events(), Lightweight, local experiment tracking for AethyxLM., JsonlExperimentTracker, Any, Path, Append-only JSONL experiment tracking with no hosted-service dependency., Write durable, machine-readable run events to a local JSONL file.

### Community 26 - "Token Embedding Layer"
Cohesion: 0.22
Nodes (7): Tensor, Token Embedding Layer for AethyxLM., Converts token IDs into dense embedding vectors., Args: token_ids: Shape -> (batch_size, sequence_length) Returns: Shape ->…, TokenEmbedding, main(), Test script for the Positional Embedding layer.

### Community 27 - "Layer Normalization"
Cohesion: 0.22
Nodes (7): LayerNorm, Tensor, Custom Layer Normalization implementation for AethyxLM. Normalizes each token…, Custom implementation of Layer Normalization. Input Shape: (batch_size,…, Args: x: Shape -> (B, T, D) Returns: Tensor: Shape -> (B, T, D), main(), Test script for the custom LayerNorm implementation.

### Community 28 - "Multilingual Corpus Preparation"
Cohesion: 0.38
Nodes (10): build_corpus(), parse_args(), Namespace, Path, Build a bounded, reproducible multilingual corpus for AethyxLM tokenizer v2.…, sample_local_text(), sha256(), stream_indic_split() (+2 more)

### Community 30 - "Scaling Law Analysis"
Cohesion: 0.31
Nodes (7): fit_scaling_law(), Fit simple empirical model/data scaling curves from AethyxLM runs., Fit L = E + A * N^-alpha * D^-beta by a floor grid search., ScalingFit, main(), Fit a scaling curve from a JSON list of training-run summaries., test_scaling_fit_recovers_synthetic_exponents()

### Community 31 - "CUDA Scaling Pilots"
Cohesion: 0.31
Nodes (9): batches(), main(), model_config(), no_grad, Run short, controlled CUDA scaling pilots on the existing token corpus. This is…, run_one(), validation_loss(), memmap (+1 more)

### Community 32 - "Weight Initialization"
Cohesion: 0.29
Nodes (6): Initialize model weights with proper initialization., init_module(), init_weights(), Module, Initialize weights for a module. Args: module: The module to initialize.…, Recursively initialize all weights in a module. Args: module: The module to…

### Community 33 - "Browser Context Compiler"
Cohesion: 0.48
Nodes (5): compileContext(), mustPreserve(), PRECISION, sha256(), terms()

### Community 34 - "Tokenizer Text Utilities"
Cohesion: 0.33
Nodes (5): preprocess(), AethyxLM Module: utils.py Purpose: Utility functions for tokenizer. Author:…, Normalize text before tokenization., Split a word into characters., split_word()

### Community 35 - "Project Dependencies"
Cohesion: 0.50
Nodes (4): AethyxLM Core Dependencies, Hugging Face Datasets, Hugging Face Tokenizers, PyTorch

### Community 36 - "Kaggle Notebook Builder"
Cohesion: 0.67
Nodes (3): build_and_save(), build_notebook(), Build and save the notebook.

### Community 38 - "Training Corpora"
Cohesion: 0.67
Nodes (3): Small AethyxLM Training Corpus, TinyStories Tokenizer Corpus, English and Indic Multilingual Training Corpus

## Knowledge Gaps
- **28 isolated node(s):** `PRECISION`, `Grouped-Query Attention`, `RoPE-Aware KV-Cached Decoding`, `Sliding-Window and Periodic Full Attention`, `ContextMemoryBank` (+23 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **8 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `AethyxTokenizer` connect `Tokenizer Model Benchmarks` to `Context Compilation Core`, `Dataset Streaming Preparation`, `Training Data Pipeline`, `Evaluation and Generation`, `Trainer Checkpoint Lifecycle`, `Interactive Chat Interface`, `Instruction Fine Tuning`, `GPT Model Compatibility`, `Training Scheduler Setup`, `Tokenizer V2 Validation`, `Training Integration Tests`?**
  _High betweenness centrality (0.235) - this node is a cross-community bridge._
- **Why does `GPT` connect `GPT Model Compatibility` to `Weight Initialization`, `Training Data Pipeline`, `Evaluation and Generation`, `Transformer Attention Blocks`, `Context Adapter Modules`, `Tokenizer Model Benchmarks`, `GPU Training Diagnostics`, `Trainer Checkpoint Lifecycle`, `Interactive Chat Interface`, `Instruction Fine Tuning`, `Training Scheduler Setup`, `Loss and Optimizer Debugging`, `Training Integration Tests`, `Data Loading Tests`, `CUDA Scaling Pilots`?**
  _High betweenness centrality (0.216) - this node is a cross-community bridge._
- **Why does `LatentContextAdapter` connect `Context Adapter Modules` to `GPT Model Compatibility`, `Context Research Contracts`?**
  _High betweenness centrality (0.067) - this node is a cross-community bridge._
- **Are the 7 inferred relationships involving `GPT` (e.g. with `ContextCrossAttention` and `LatentContextAdapter`) actually correct?**
  _`GPT` has 7 INFERRED edges - model-reasoned connections that need verification._
- **Are the 6 inferred relationships involving `AethyxTokenizer` (e.g. with `AethyxDataset` and `MixedAethyxDataset`) actually correct?**
  _`AethyxTokenizer` has 6 INFERRED edges - model-reasoned connections that need verification._
- **Are the 6 inferred relationships involving `Trainer` (e.g. with `StateStub` and `TrainingBatches`) actually correct?**
  _`Trainer` has 6 INFERRED edges - model-reasoned connections that need verification._
- **What connects `PRECISION`, `Grouped-Query Attention`, `RoPE-Aware KV-Cached Decoding` to the rest of the system?**
  _28 weakly-connected nodes found - possible documentation gaps or missing edges._