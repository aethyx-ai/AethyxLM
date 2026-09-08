# AethyxLM Architecture

This document describes the current v3 model and inference runtime. Executable
code and checkpoint metadata remain authoritative when this document and a
checkpoint disagree.

## Current v3 model

The current research configuration is a decoder-only transformer with roughly
137.6 million parameters:

| Component | v3 configuration |
|---|---:|
| Vocabulary | 48,000 tokens (tokenizer v3) |
| Transformer layers | 16 |
| Hidden dimension | 768 |
| Query heads | 12 |
| Key/value heads | 4 |
| Feed-forward dimension | 2,048 |
| Model context | 1,024 tokens |
| Normalization | RMSNorm |
| Position encoding | RoPE |
| Feed-forward block | SwiGLU |
| Attention implementation | PyTorch SDPA |
| Projection layout | Fused QKV |
| Embedding/output weights | Tied |

Grouped-query attention reduces the number of cached key/value heads from 12
to 4. QK normalization controls attention-vector scale. Bias-free projections,
RMSNorm and SwiGLU provide the current modern baseline. RoPE tables can be
allocated beyond 1,024 positions, but this does not establish reliable
understanding beyond the model's evaluated context length.

Legacy checkpoint loading reconstructs architectural values from checkpoint
tensors and metadata. It removes obsolete persistent causal masks while still
requiring state-dictionary compatibility.

## Inference architecture

### Prompt contracts

`inference/prompt_contract.py` is the canonical source for serialization used
by interactive inference, evaluation and supervised fine-tuning data.

Three versioned contracts are supported:

- `base-v1` sends raw text for continuation.
- `legacy-chat-v1` uses `User:` and `Aethyx:` text markers.
- `aethyx-sft-v1` uses `<SYSTEM>`, `<USER>` and `<ASSISTANT>` role tokens.

Checkpoints may declare `config.inference.prompt_contract`. Automatic mode only
selects chat behavior when that metadata explicitly declares a chat contract.
A metadata-free checkpoint defaults to base completion. This prevents a base
model from being silently prompted with an instruction format it never
declared.

History trimming removes complete oldest turns instead of cutting arbitrary
token fragments. For the v3 SFT contract, the initial system segment is
protected when it fits the available budget. Generation results report prompt
token count, discarded prompt tokens and the finish reason.

### Logit projection

`GPT.forward()` accepts `logits_mode="all"` for callers that require a logit at
every position and `logits_mode="last"` for generation. The latter sends only
the final hidden state through the 48,000-way output projection during prompt
prefill and cached decoding.

On the local 246K v3 checkpoint, the bounded CPU prefill measurement showed
approximately 1.28x speedup for final-position projection. This is a short host
measurement and should be repeated on the deployment GPU before being treated
as a production performance figure.

### KV caching and grouped-query attention

The attention module supports the original tuple cache and a preallocated
cache. The preallocated form reserves capacity and writes new key/value states
into indexed slices, avoiding a full `torch.cat` copy for every decoded token.

Automatic cache selection currently uses:

- tuple caching on CPU, where the bounded local test found preallocation slower;
- preallocated caching on CUDA, subject to deployment benchmarking;
- the tuple representation as a compatibility fallback.

The CUDA path can call SDPA's native grouped-query attention without expanding
four KV heads to twelve. If the active backend exposes the API but rejects the
operation, attention disables the native path and falls back to expanded KV
heads.

Generation reserves an explicit response budget. The default overflow policy
stops with `context_limit` rather than repeatedly recomputing a full 1,024-token
window for every additional token. The reference `recompute` policy remains
available for compatibility. Dropping cached tokens and rebasing RoPE positions
is deliberately avoided because it would not be numerically equivalent to the
original sequence.

### Decoding profiles

The generation runtime provides named sampling profiles for prose, code and
exact extraction, alongside greedy decoding. Repetition penalties and n-gram
blocking now inspect generated output rather than the source prompt. This
allows answers to copy identifiers, numbers, code fragments and evidence from
the prompt without being penalized merely because those tokens already appear
in the input.

Batch generation groups equal-length prompts and uses last-position logits and
the same cache and overflow rules as single-prompt generation.

## Retrieval

`inference/retrieval.py` supplies a small dependency-free local evidence index.
It uses BM25-style lexical scoring, assigns every passage a stable source ID and
returns selected passages verbatim. Evidence formatting admits complete
passages within a token budget instead of truncating names, identifiers or
numeric values halfway through a passage.

The chat interface can load local evidence, retrieve a bounded number of
passages and insert source-labelled evidence into the canonical prompt. This is
a local retrieval baseline; it does not prove that a generated answer is
entailed by a cited passage.

## Tools

`inference/tools.py` implements a strict tool-call envelope:

```json
{"tool": "calculator", "arguments": {"expression": "17 * 23"}}
```

The calculator parses a bounded arithmetic grammar with Python's AST and
`Decimal`. It does not use `eval`, accept names or permit function calls.
When calculator tools are enabled, the interactive interface detects clear
standalone arithmetic such as `What is 17 x 23?` and returns the validated
result directly. The detector uses an anchored grammar and falls through to
normal generation for prose, variables, units, dates, identifiers and invalid
operations. Explicit `/tool` calculator requests remain available.

A restricted Python backend is available behind an explicit command-line flag.
It applies an AST allowlist, blocks imports and attribute access, exposes only a
small set of pure built-ins, uses an isolated temporary directory and clean
environment, and limits execution time and output. POSIX systems additionally
enforce address-space, CPU and process limits. Windows does not provide the
same memory/process isolation through this backend, so code execution remains
disabled by default and should not be treated as a strong security boundary on
Windows.

Tool requests use an exact JSON schema. Malformed requests have bounded retry
handling, and structured results are reinserted through the active prompt
contract for the model's final response. Schema validity only establishes that
a request is well formed; it does not establish that the selected tool or
arguments are semantically correct.

## Quantization

`inference/quantization.py` keeps quantization opt-in and CPU-only. It supports
dynamic INT8 for feed-forward layers or all eligible linear layers and reports
actual float and quantized module coverage. Custom Aethyx `Linear` subclasses
are converted into modules supported by the PyTorch dynamic quantizer on a
copied model.

Bounded measurements on the local 246K checkpoint produced:

| Mode | Serialized state size | Result |
|---|---:|---|
| Unquantized | approximately 550 MB | Reference |
| Dynamic INT8, FFN only | approximately 324 MB | Severe numerical drift |
| Dynamic INT8, full | approximately 285 MB | Severe numerical drift |

The current INT8 modes reduce storage but are not recommended for this
checkpoint because the measured output drift is unacceptable. No GPU
quantization performance claim has been established.

## Benchmarks and verification

`benchmark_inference.py` measures full versus last-position projection and
tuple versus preallocated decoding. `benchmark_quantization.py` measures
latency, serialized size, module coverage, logit error, cosine similarity and
top-1 agreement. Both scripts deliberately use short bounded runs.

The public model, inference, evaluation and SFT suite passed 151 tests. The
bounded local verification used CPU execution; deployment CUDA performance
remains to be measured.

## Context direction

AethyxLM's longer-term direction includes a private context representation and
compilation layer intended to reduce redundant context while preserving useful
information. The current transformer remains a stable language-model baseline,
and claims about compression, longer effective context or retrieval quality
require benchmark evidence before they are treated as model capabilities.

## References

- [RMSNorm](https://arxiv.org/abs/1910.07467)
- [RoPE](https://arxiv.org/abs/2104.09864)
- [SwiGLU](https://arxiv.org/abs/2002.05202)
- [PyTorch scaled dot-product attention](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
