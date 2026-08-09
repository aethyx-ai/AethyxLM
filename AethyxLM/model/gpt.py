"""
Complete GPT model for AethyxLM.
"""

import torch
import torch.nn as nn

from model.config import (
    VOCAB_SIZE,
    CONTEXT_LENGTH,
    EMBED_DIM,
    NUM_HEADS,
    NUM_LAYERS,
    FFN_DIM,
    DROPOUT,
    USE_BIAS,
    LAYER_NORM_EPS,
    NORMALIZATION,
    POSITION_ENCODING,
    FFN_TYPE,
    ROPE_BASE,
    ROPE_MAX_SEQ_LEN,
)


class GPT(nn.Module):
    """
    Decoder-only GPT Language Model.
    """

    def __init__(self, vocab_size: int = None, config: dict = None):
        super().__init__()

        # Merge defaults from config.py with overrides from config dict
        if config is None:
            config = {}

        self.vocab_size = vocab_size if vocab_size is not None else config.get('vocab_size', VOCAB_SIZE)
        self.context_length = config.get('context_length', CONTEXT_LENGTH)
        self.embed_dim = config.get('embed_dim', EMBED_DIM)
        self.num_heads = config.get('num_heads', NUM_HEADS)
        self.num_kv_heads = config.get('num_kv_heads', self.num_heads)
        self.num_layers = config.get('num_layers', NUM_LAYERS)
        self.ffn_dim = config.get('ffn_dim', FFN_DIM)
        self.dropout_rate = config.get('dropout', DROPOUT)
        self.use_bias = config.get('use_bias', USE_BIAS)
        self.layer_norm_eps = config.get('layer_norm_eps', LAYER_NORM_EPS)

        # Architecture options
        self.normalization = config.get('normalization', NORMALIZATION)
        self.position_encoding = config.get('position_encoding', POSITION_ENCODING)
        self.ffn_type = config.get('ffn_type', FFN_TYPE)
        self.rope_base = config.get('rope_base', 10000.0)
        self.rope_max_seq_len = config.get('rope_max_seq_len', 8192)
        self.rope_scaling_factor = config.get('rope_scaling_factor', 1.0)
        self.fused_qkv = config.get('fused_qkv', False)
        self.use_sdpa = config.get('use_sdpa', True)
        self.qk_norm = config.get('qk_norm', False)
        self.gradient_checkpointing = config.get('gradient_checkpointing', False)
        self.context_adapter_config = config.get('context_adapter', {})
        self.sliding_window = config.get('sliding_window')
        self.global_attention_interval = config.get('global_attention_interval', 0)

        # ----------------------------------------
        # Token Embedding
        # ----------------------------------------

        self.token_embedding = nn.Embedding(self.vocab_size, self.embed_dim)

        # ----------------------------------------
        # Positional Embedding (only for learned)
        # ----------------------------------------

        if self.position_encoding == "learned":
            self.position_embedding = nn.Embedding(self.context_length, self.embed_dim)
        else:
            self.position_embedding = None

        # ----------------------------------------
        # Dropout
        # ----------------------------------------

        self.dropout = nn.Dropout(self.dropout_rate)

        # ----------------------------------------
        # Transformer Blocks
        # ----------------------------------------

        from model.transformer_block import TransformerBlock
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim=self.embed_dim,
                    num_heads=self.num_heads,
                    num_kv_heads=self.num_kv_heads,
                    ffn_dim=self.ffn_dim,
                    dropout=self.dropout_rate,
                    context_length=self.context_length,
                    use_bias=self.use_bias,
                    layer_norm_eps=self.layer_norm_eps,
                    normalization=self.normalization,
                    ffn_type=self.ffn_type,
                    position_encoding=self.position_encoding,
                    rope_base=self.rope_base,
                    rope_max_seq_len=self.rope_max_seq_len,
                    rope_scaling_factor=self.rope_scaling_factor,
                    fused_qkv=self.fused_qkv,
                    use_sdpa=self.use_sdpa,
                    qk_norm=self.qk_norm,
                    sliding_window=(
                        None
                        if self.global_attention_interval
                        and (layer_index + 1) % self.global_attention_interval == 0
                        else self.sliding_window
                    ),
                )
                for layer_index in range(self.num_layers)
            ]
        )

        self.context_adapter = None
        self.context_cross_attention = nn.ModuleDict()
        if self.context_adapter_config.get("enabled", False):
            from model.context_adapter import ContextCrossAttention, LatentContextAdapter

            adapter_heads = self.context_adapter_config.get("num_heads", self.num_heads)
            self.context_adapter = LatentContextAdapter(
                input_dim=self.context_adapter_config.get("input_dim", self.embed_dim),
                embed_dim=self.embed_dim,
                num_latents=self.context_adapter_config.get("num_latents", 64),
                num_heads=adapter_heads,
                num_types=self.context_adapter_config.get("num_types", 16),
                depth=self.context_adapter_config.get("depth", 2),
                dropout=self.dropout_rate,
            )
            cross_layers = self.context_adapter_config.get(
                "cross_attention_layers", [self.num_layers - 1]
            )
            for layer_index in cross_layers:
                if not 0 <= int(layer_index) < self.num_layers:
                    raise ValueError("context cross-attention layer index is out of range")
                self.context_cross_attention[str(int(layer_index))] = ContextCrossAttention(
                    self.embed_dim, adapter_heads, self.dropout_rate
                )

        # ----------------------------------------
        # Final Layer Normalization
        # ----------------------------------------

        from model.modules.rmsnorm import build_normalization
        self.final_norm = build_normalization(
            embed_dim=self.embed_dim,
            normalization=self.normalization,
            eps=self.layer_norm_eps,
        )

        # ----------------------------------------
        # Language Modeling Head
        # ----------------------------------------

        self.lm_head = nn.Linear(
            self.embed_dim,
            self.vocab_size,
            bias=False,
        )

        # Share embedding weights with output layer
        self.lm_head.weight = self.token_embedding.weight

        # Initialize weights
        self._init_weights()
        self._print_architecture_summary()

    def _init_weights(self):
        """Initialize model weights with proper initialization."""
        from model.layers import init_module
        import math
        init_module(self, init_type="normal", init_std=0.02)

        # Special initialization for embeddings
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        if self.position_embedding is not None:
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        # Special initialization for transformer blocks with scaled initialization
        for layer in self.layers:
            # Attention projections
            if layer.attention.fused_qkv:
                nn.init.normal_(layer.attention.qkv_proj.weight, mean=0.0, std=0.02)
            else:
                nn.init.normal_(layer.attention.q_proj.weight, mean=0.0, std=0.02)
                nn.init.normal_(layer.attention.k_proj.weight, mean=0.0, std=0.02)
                nn.init.normal_(layer.attention.v_proj.weight, mean=0.0, std=0.02)
            nn.init.normal_(layer.attention.out_proj.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.num_layers))

            # FFN layers
            if hasattr(layer.feed_forward, 'fc1') and hasattr(layer.feed_forward, 'fc2'):
                nn.init.normal_(layer.feed_forward.fc1.weight, mean=0.0, std=0.02)
                nn.init.normal_(layer.feed_forward.fc2.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.num_layers))

        # Output projection (already tied to token_embedding, so no additional init needed)

    def _print_architecture_summary(self):
        """Print architecture configuration at initialization."""
        print("=" * 60)
        print("AethyxLM Architecture Summary")
        print("=" * 60)
        print(f"Normalization:       {self.normalization.upper()}")
        print(f"Position Encoding:   {self.position_encoding.upper()}")
        print(f"Feed Forward:        {self.ffn_type.upper()}")
        print(f"Attention:           {'RoPE' if self.position_encoding == 'rope' else 'Learned PosEmb'}")
        print(f"Attention heads:     {self.num_heads} Q / {self.num_kv_heads} KV")
        print(f"Attention kernel:    {'SDPA' if self.use_sdpa else 'manual'}")
        print(f"Fused QKV:           {self.fused_qkv}")
        print(f"Parameters:          {sum(p.numel() for p in self.parameters()):,}")
        print("=" * 60)

    def encode_context(self, context_features, context_type_ids=None, context_mask=None):
        if self.context_adapter is None:
            raise RuntimeError("context_adapter is not enabled in this model")
        return self.context_adapter(context_features, context_type_ids, context_mask)

    def load_compatible_state_dict(self, state_dict, strict: bool = True):
        """Load current and legacy checkpoints without persisting causal masks."""
        cleaned = {
            key: value
            for key, value in state_dict.items()
            if not key.endswith(".causal_mask") and key != "causal_mask"
        }
        return self.load_state_dict(cleaned, strict=strict)

    def forward(
        self,
        input_ids: torch.Tensor,
        kv_cache=None,
        use_cache: bool = False,
        context_features=None,
        context_type_ids=None,
        context_mask=None,
        context_latents=None,
    ):
        """
        Forward pass.

        Args:
            input_ids:
                Shape -> (batch_size, context_length)

        Returns:
            logits:
                Shape -> (batch_size, context_length, vocab_size)
        """

        # ----------------------------------------
        # Token Embeddings
        # ----------------------------------------

        x = self.token_embedding(input_ids)

        # ----------------------------------------
        # Add Positional Embeddings (if learned)
        # ----------------------------------------

        past_length = 0
        if kv_cache:
            first_cache = kv_cache[0]
            past_length = (
                int(first_cache[2]) if len(first_cache) > 2 else first_cache[0].size(2)
            )

        if self.position_encoding == "learned":
            seq_len = input_ids.size(1)
            if past_length + seq_len > self.context_length:
                raise ValueError("learned positional embeddings exceeded context_length")
            positions = torch.arange(
                past_length, past_length + seq_len, device=input_ids.device, dtype=torch.long
            )
            x = x + self.position_embedding(positions)

        x = self.dropout(x)

        if context_features is not None and context_latents is not None:
            raise ValueError("provide context_features or context_latents, not both")
        if context_features is not None:
            context_latents = self.encode_context(
                context_features, context_type_ids, context_mask
            )
        if context_latents is not None and self.context_adapter is None:
            raise RuntimeError("context latents require an enabled context_adapter")

        # ----------------------------------------
        # Transformer Blocks
        # ----------------------------------------

        presents = []
        for index, layer in enumerate(self.layers):
            layer_cache = None if kv_cache is None else kv_cache[index]
            if use_cache:
                x, present = layer(x, kv_cache=layer_cache, use_cache=True)
                presents.append(present)
            else:
                if self.gradient_checkpointing and self.training:
                    from torch.utils.checkpoint import checkpoint
                    x = checkpoint(layer, x, use_reentrant=False)
                else:
                    x = layer(x)
            layer_key = str(index)
            if context_latents is not None and layer_key in self.context_cross_attention:
                x = self.context_cross_attention[layer_key](x, context_latents)

        # ----------------------------------------
        # Final LayerNorm
        # ----------------------------------------

        x = self.final_norm(x)

        # ----------------------------------------
        # Project to Vocabulary
        # ----------------------------------------

        logits = self.lm_head(x)

        return (logits, presents) if use_cache else logits

    @classmethod
    def _infer_checkpoint_config(cls, state_dict: dict, metadata: dict) -> dict:
        """Reconstruct legacy model configuration from authoritative tensors."""
        config = metadata.copy()
        embedding = state_dict["token_embedding.weight"]
        config["vocab_size"], config["embed_dim"] = embedding.shape

        if "position_embedding.weight" in state_dict:
            config["position_encoding"] = "learned"
            config["context_length"] = state_dict["position_embedding.weight"].shape[0]
        else:
            config.setdefault("position_encoding", "rope")

        layer_indices = {
            int(key.split(".")[1])
            for key in state_dict
            if key.startswith("layers.") and key.split(".")[1].isdigit()
        }
        if layer_indices:
            config["num_layers"] = max(layer_indices) + 1

        first_prefix = "layers.0."
        config["normalization"] = (
            "layernorm"
            if first_prefix + "norm1.gamma" in state_dict
            else "rmsnorm"
        )
        config["fused_qkv"] = first_prefix + "attention.qkv_proj.weight" in state_dict
        config["use_bias"] = any(
            key.endswith(".bias") for key in state_dict if key.startswith(first_prefix)
        )
        config.setdefault("num_heads", NUM_HEADS)

        if config["fused_qkv"]:
            qkv_rows = state_dict[first_prefix + "attention.qkv_proj.weight"].shape[0]
            kv_dim = (qkv_rows - config["embed_dim"]) // 2
            head_dim = config["embed_dim"] // config["num_heads"]
            config["num_kv_heads"] = kv_dim // head_dim
        else:
            key_rows = state_dict[first_prefix + "attention.k_proj.weight"].shape[0]
            head_dim = config["embed_dim"] // config["num_heads"]
            config["num_kv_heads"] = key_rows // head_dim

        fc1 = state_dict.get(first_prefix + "feed_forward.fc1.weight")
        fc2 = state_dict.get(first_prefix + "feed_forward.fc2.weight")
        if fc1 is not None and fc2 is not None:
            config["ffn_dim"] = fc2.shape[1]
            config["ffn_type"] = (
                "swiglu" if fc1.shape[0] == 2 * fc2.shape[1] else "gelu"
            )
        config.setdefault("dropout", DROPOUT)
        config.setdefault("layer_norm_eps", LAYER_NORM_EPS)
        return config

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, device: str = "cpu", config: dict = None):
        """
        Load model from checkpoint with automatic architecture detection.

        Args:
            checkpoint_path: Path to checkpoint file.
            device: Device to load model on.
            config: Optional config dict to override checkpoint config.

        Returns:
            Loaded GPT model.

        Raises:
            RuntimeError: If checkpoint architecture is incompatible.
        """
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # Extract config from checkpoint
        ckpt_config = checkpoint.get("config", {})
        metadata_config = ckpt_config.get("model", ckpt_config)
        model_config = cls._infer_checkpoint_config(
            checkpoint["model_state_dict"], metadata_config
        )

        # Determine architecture from checkpoint
        arch_config = {
            "normalization": model_config.get("normalization", "layernorm"),
            "position_encoding": model_config.get("position_encoding", "learned"),
            "ffn_type": model_config.get("ffn_type", "gelu"),
        }

        # Override with provided config if given
        if config:
            requested = config.get("model", config).copy()
            requested.update(
                {
                    "vocab_size": model_config["vocab_size"],
                    "embed_dim": model_config["embed_dim"],
                    "num_layers": model_config["num_layers"],
                }
            )
            model_config = requested

        # Check architecture compatibility
        if config:
            user_model_config = config.get("model", config)
            for key in ["normalization", "position_encoding", "ffn_type"]:
                if key in user_model_config and user_model_config[key] != arch_config[key]:
                    raise RuntimeError(
                        f"{key} mismatch: checkpoint has '{arch_config[key]}', "
                        f"but config requests '{user_model_config[key]}'"
                    )

        # Create model with detected architecture
        model = cls(vocab_size=model_config.get("vocab_size"), config=model_config)
        model.to(device)

        # Load state dict
        model.load_compatible_state_dict(checkpoint["model_state_dict"], strict=True)

        return model
