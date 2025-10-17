import torch
import torch.nn as nn
import math
from typing import Optional


class PositionalEncoding(nn.Module):
    """3D + Temporal positional encoding"""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model

        # Standard sinusoidal encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, D]
        return x + self.pe[:x.size(1), :].unsqueeze(0)


class TraceEncoder(nn.Module):
    """Encode trace sequences with Transformer"""

    def __init__(self, d_model: int = 128, nhead: int = 4,
                 num_layers: int = 3, dim_feedforward: int = 512):
        super().__init__()

        # Input projection: (x,y,z,t) -> d_model
        self.input_proj = nn.Linear(4, d_model)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

    def forward(self, traces: torch.Tensor, mask: Optional[torch.Tensor] = None):
        # traces: [B, N, 4]
        # mask: [B, N] - True for valid positions

        # Project input
        x = self.input_proj(traces)  # [B, N, D]

        # Add positional encoding
        x = self.pos_encoding(x)

        # Create attention mask (inverted for transformer)
        if mask is not None:
            attn_mask = ~mask  # [B, N] - True means ignore
        else:
            attn_mask = None

        # Encode
        encoded = self.transformer(x, src_key_padding_mask=attn_mask)

        return encoded


class ColliderDecoder(nn.Module):
    """Decode colliders using learnable queries (DETR-style)"""

    def __init__(self, d_model: int = 128, nhead: int = 4,
                 num_layers: int = 3, num_queries: int = 30):
        super().__init__()

        self.num_queries = num_queries

        # Learnable object queries
        self.query_embed = nn.Embedding(num_queries, d_model)

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=512,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers)

        # Prediction heads
        self.box_head = MLP(d_model, d_model, 6, 2)  # cx,cy,cz,sx,sy,sz
        self.class_head = nn.Linear(d_model, 4)  # BLOCK/LOW/MID/HIGH

    def forward(self, memory: torch.Tensor, memory_mask: Optional[torch.Tensor] = None):
        # memory: [B, N, D] - encoded traces
        # memory_mask: [B, N]

        B = memory.shape[0]

        # Get queries
        queries = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)  # [B, Q, D]

        # Decode
        if memory_mask is not None:
            mem_mask = ~memory_mask
        else:
            mem_mask = None

        decoded = self.transformer(
            queries,
            memory,
            memory_key_padding_mask=mem_mask
        )  # [B, Q, D]

        # Predict boxes and classes
        boxes = self.box_head(decoded)  # [B, Q, 6]
        classes = self.class_head(decoded)  # [B, Q, 4]

        return boxes, classes


class MLP(nn.Module):
    """Simple MLP"""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int):
        super().__init__()

        layers = []
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            out_dim = output_dim if i == num_layers - 1 else hidden_dim

            layers.append(nn.Linear(in_dim, out_dim))
            if i < num_layers - 1:
                layers.append(nn.ReLU())

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class TraceToColliderTransformer(nn.Module):
    """Complete model: Trace -> Colliders (Lightweight version for M4 24GB)"""

    def __init__(self, d_model: int = 128, nhead: int = 4,
                 num_encoder_layers: int = 3, num_decoder_layers: int = 3,
                 num_queries: int = 30):
        super().__init__()

        self.encoder = TraceEncoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_encoder_layers
        )

        self.decoder = ColliderDecoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_decoder_layers,
            num_queries=num_queries
        )

    def forward(self, traces: torch.Tensor, mask: Optional[torch.Tensor] = None):
        # Encode traces
        encoded = self.encoder(traces, mask)

        # Decode colliders
        boxes, classes = self.decoder(encoded, mask)

        return {
            'pred_boxes': boxes,  # [B, Q, 6]
            'pred_classes': classes  # [B, Q, 4]
        }


def build_model(num_queries: int = 30, d_model: int = 128):
    """
    Build lightweight model for M4 24GB.

    Reduced parameters:
    - d_model: 256 -> 128
    - num_encoder_layers: 6 -> 3
    - num_decoder_layers: 6 -> 3
    - nhead: 8 -> 4
    - dim_feedforward: 1024 -> 512
    - num_queries: 50 -> 30

    Args:
        num_queries: Number of object queries (max detections per scene)
        d_model: Model dimension

    Returns:
        TraceToColliderTransformer model
    """
    model = TraceToColliderTransformer(
        d_model=d_model,
        nhead=4,
        num_encoder_layers=3,
        num_decoder_layers=3,
        num_queries=num_queries
    )

    return model


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test model
    print("Building lightweight model...")
    model = build_model()

    # Count parameters
    num_params = count_parameters(model)
    print(f"Total trainable parameters: {num_params:,}")
    print(f"Estimated model size: ~{num_params * 4 / 1024 / 1024:.1f} MB (FP32)")

    # Test forward pass
    print("\nTesting forward pass...")
    batch_size = 2
    seq_len = 1000
    traces = torch.randn(batch_size, seq_len, 4)
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

    output = model(traces, mask)

    print("Output shapes:")
    print(f"  Boxes: {output['pred_boxes'].shape}")
    print(f"  Classes: {output['pred_classes'].shape}")

    # Memory estimate
    print(f"\nMemory estimate for batch_size={batch_size}, seq_len={seq_len}:")
    print(f"  Input: ~{batch_size * seq_len * 4 * 4 / 1024 / 1024:.2f} MB")
    print(f"  Model: ~{num_params * 4 / 1024 / 1024:.1f} MB")
    print(f"  Output: ~{batch_size * 30 * 10 * 4 / 1024:.2f} MB")

    print("\n✓ Model test passed!")