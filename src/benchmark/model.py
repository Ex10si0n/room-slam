import torch
import torch.nn as nn
import math
from typing import Optional


class PositionalEncoding(nn.Module):
    """3D + Temporal positional encoding with dynamic length support"""

    def __init__(self, d_model: int, max_len: int = 20000):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

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
        seq_len = x.size(1)

        # If sequence is longer than max_len, extend positional encoding
        if seq_len > self.max_len:
            self._extend_pe(seq_len, x.device)

        return x + self.pe[:seq_len, :].unsqueeze(0)

    def _extend_pe(self, new_len: int, device):
        """Dynamically extend positional encoding if needed"""
        pe = torch.zeros(new_len, self.d_model, device=device)
        position = torch.arange(0, new_len, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2, device=device).float() *
                             (-math.log(10000.0) / self.d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        self.max_len = new_len


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
        """
        Args:
            traces: [B, N, 4] with (x,y,z,t)
            mask:   [B, N] boolean; True means valid token

        Returns:
            encoded: [B, N, D] transformer features
            coords:  [B, N, 3] raw (x,y,z) coordinates (not encoded)
        """
        # Keep raw coordinates for anchor computation
        coords = traces[..., :3].contiguous()  # [B, N, 3]

        # Project input to model dimension
        x = self.input_proj(traces)  # [B, N, D]

        # Add positional encoding
        x = self.pos_encoding(x)

        # Create attention mask (True -> ignore)
        attn_mask = ~mask if mask is not None else None

        # Encode traces
        encoded = self.transformer(x, src_key_padding_mask=attn_mask)  # [B,N,D]

        return encoded, coords

class ColliderDecoder(nn.Module):
    """
    Decode colliders using learnable queries (DETR-style), predicting boxes
    relative to attention-weighted anchors on the input trace coordinates.
    """

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

        # Heads:
        # - delta for center offset relative to the anchor
        # - size logit will be passed through softplus to ensure positive sizes
        self.center_delta_head = MLP(d_model, d_model, 3, 2)  # Δcx, Δcy, Δcz
        self.size_head = MLP(d_model, d_model, 3, 2)          # raw logits -> softplus
        self.class_head = nn.Linear(d_model, 4)               # BLOCK/LOW/MID/HIGH

        # Lightweight attention projections for anchor computation
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.scale = d_model ** 0.5  # for dot-product attention scaling

    def forward(
        self,
        memory: torch.Tensor,               # [B, N, D] encoded features
        coords: torch.Tensor,               # [B, N, 3] raw (x,y,z) of traces
        memory_mask: Optional[torch.Tensor] = None  # [B, N] True for valid
    ):
        """
        Returns:
            boxes:   [B, Q, 6] absolute boxes (cx,cy,cz,sx,sy,sz)
            classes: [B, Q, 4] class logits
        """
        B, N, D = memory.shape

        # Prepare queries
        queries = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)  # [B, Q, D]

        # Key padding mask for transformer: True means to ignore
        mem_pad_mask = ~memory_mask if memory_mask is not None else None

        # Decode query features with cross-attention over the memory
        decoded = self.transformer(
            queries,
            memory,
            memory_key_padding_mask=mem_pad_mask
        )  # [B, Q, D]

        # ---------- Anchor attention over raw coordinates ----------
        # Compute attention weights from decoded queries to memory tokens.
        # We do a simple dot-product attention with separate projections.
        q = self.q_proj(decoded)                 # [B, Q, D]
        k = self.k_proj(memory)                  # [B, N, D]
        attn_scores = torch.einsum('bqd,bnd->bqn', q, k) / self.scale  # [B, Q, N]

        if memory_mask is not None:
            # Mask out padded positions: set them to a very negative value
            # memory_mask: True for valid -> we want False for padded, so invert:
            pad = ~memory_mask  # True where padded
            attn_scores = attn_scores.masked_fill(pad.unsqueeze(1), float('-inf'))

        attn_weights = torch.softmax(attn_scores, dim=-1)  # [B, Q, N]

        # Anchor position is the attention-weighted average of raw coords
        # coords: [B, N, 3], attn: [B, Q, N] -> anchor: [B, Q, 3]
        anchor_pos = torch.einsum('bqn,bnd->bqd', attn_weights, coords)

        # ---------- Predict relative centers and sizes ----------
        delta_center = self.center_delta_head(decoded)       # [B, Q, 3]
        size_raw = self.size_head(decoded)                   # [B, Q, 3]

        # Absolute center = anchor + delta
        center = anchor_pos + delta_center                   # [B, Q, 3]

        # Positive sizes via softplus; add epsilon to avoid zeros
        size = torch.nn.functional.softplus(size_raw) + 1e-4 # [B, Q, 3]

        # Pack boxes: (cx,cy,cz,sx,sy,sz)
        boxes = torch.cat([center, size], dim=-1)            # [B, Q, 6]

        # Class logits from decoded features
        classes = self.class_head(decoded)                   # [B, Q, 4]

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
    """Complete model: Trace -> Colliders (relative centers to anchors)"""

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
        # Encode traces -> features and raw coordinates
        memory, coords = self.encoder(traces, mask)  # memory:[B,N,D], coords:[B,N,3]

        # Decode colliders with anchor-relative prediction
        boxes, classes = self.decoder(memory, coords, mask)

        return {
            'pred_boxes': boxes,     # [B, Q, 6] absolute boxes
            'pred_classes': classes  # [B, Q, 4] class logits
        }

def build_model(num_queries: int = 60, d_model: int = 256):
    """
    Build a larger model for higher capacity.

    These bumps increase both representation power and the maximum #detections.
    """
    model = TraceToColliderTransformer(
        d_model=d_model,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        num_queries=num_queries
    )
    return model


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Setup device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device("cpu")
        print("CUDA not available, using CPU")

    # Test model
    print("\nBuilding lightweight model...")
    model = build_model().to(device)

    # Count parameters
    num_params = count_parameters(model)
    print(f"Total trainable parameters: {num_params:,}")
    print(f"Estimated model size: ~{num_params * 4 / 1024 / 1024:.1f} MB (FP32)")

    # Test forward pass
    print("\nTesting forward pass...")
    batch_size = 2
    seq_len = 1000
    traces = torch.randn(batch_size, seq_len, 4).to(device)
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool).to(device)

    output = model(traces, mask)

    print("Output shapes:")
    print(f"  Boxes: {output['pred_boxes'].shape}")
    print(f"  Classes: {output['pred_classes'].shape}")

    # Memory estimate
    print(f"\nMemory estimate for batch_size={batch_size}, seq_len={seq_len}:")
    print(f"  Input: ~{batch_size * seq_len * 4 * 4 / 1024 / 1024:.2f} MB")
    print(f"  Model: ~{num_params * 4 / 1024 / 1024:.1f} MB")
    print(f"  Output: ~{batch_size * 30 * 10 * 4 / 1024:.2f} MB")

    if torch.cuda.is_available():
        print(f"\nGPU Memory allocated: {torch.cuda.memory_allocated(device) / 1024 / 1024:.2f} MB")
        print(f"GPU Memory reserved: {torch.cuda.memory_reserved(device) / 1024 / 1024:.2f} MB")

    print("\n✓ Model test passed!")