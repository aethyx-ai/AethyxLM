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

        # Print architecture summary
        self._print_architecture_summary()

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
                    ffn_dim=self.ffn_dim,
                    dropout=self.dropout_rate,
                    context_length=self.context_length,
                    use_bias=self.use_bias,
                    layer_norm_eps=self.layer_norm_eps,
                    normalization=self.normalization,
                    ffn_type=self.ffn_type,
                )
                for _ in range(self.num_layers)
            ]
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
        print(f"Parameters:          {sum(p.numel() for p in self.parameters()):,}")
        print("=" * 60)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
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

        if self.position_encoding == "learned":
            seq_len = input_ids.size(1)
            positions = torch.arange(seq_len, device=input_ids.device, dtype=torch.long)
            x = x + self.position_embedding(positions)

        x = self.dropout(x)

        # ----------------------------------------
        # Transformer Blocks
        # ----------------------------------------

        for layer in self.layers:
            x = layer(x)

        # ----------------------------------------
        # Final LayerNorm
        # ----------------------------------------

        x = self.final_norm(x)

        # ----------------------------------------
        # Project to Vocabulary
        # ----------------------------------------

        logits = self.lm_head(x)

        return logits

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
        model_config = ckpt_config.get("model", {})

        # Determine architecture from checkpoint
        arch_config = {
            "normalization": model_config.get("normalization", "layernorm"),
            "position_encoding": model_config.get("position_encoding", "learned"),
            "ffn_type": model_config.get("ffn_type", "gelu"),
        }

        # Override with provided config if given
        if config:
            model_config = config
        else:
            model_config = ckpt_config.get("model", {})

        # Check architecture compatibility
        if config and "model" in config:
            user_model_config = config["model"]
            for key in ["normalization", "position_encoding", "ffn_type"]:
                if key in user_model_config and user_model_config[key] != arch_config[key]:
                    raise RuntimeError(
                        f"{key} mismatch: checkpoint has '{arch_config[key]}', "
                        f"but config requests '{user_model_config[key]}'"
                    )

        # Ensure vocab_size from checkpoint is preserved unless explicitly overridden
        if "vocab_size" not in model_config:
            model_config["vocab_size"] = ckpt_config.get("model", {}).get("vocab_size", VOCAB_SIZE)

        # Create model with detected architecture
        model = cls(vocab_size=model_config.get("vocab_size"), config=model_config)
        model.to(device)

        # Load state dict
        missing_keys, unexpected_keys = model.load_state_dict(
            checkpoint["model_state_dict"], strict=False
        )

        if missing_keys:
            print(f"Warning: Missing keys in checkpoint: {missing_keys}")
        if unexpected_keys:
            print(f"Warning: Unexpected keys in checkpoint: {unexpected_keys}")

        return model