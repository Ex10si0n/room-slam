import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class PositionalEncoding(nn.Module):
    """1D positional encoding for variable-length trace sequences."""

    def __init__(self, d_model: int, max_len: int = 20000):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, D] input features
        Returns:
            x + positional encoding (broadcasted on batch)
        """
        seq_len = x.size(1)
        if seq_len > self.max_len:
            self._extend_pe(seq_len, x.device)
        return x + self.pe[:seq_len, :].unsqueeze(0)

    def _extend_pe(self, new_len: int, device):
        """Extend positional encoding if sequence is longer than max_len."""
        pe = torch.zeros(new_len, self.d_model, device=device)
        position = torch.arange(0, new_len, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2, device=device).float() *
                             (-math.log(10000.0) / self.d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)


# -----------------------------
# Trace encoder (Transformer)
# -----------------------------
class TransformerTraceEncoder(nn.Module):
    """
    Transformer encoder for trace sequences.

    Input:
        traces: [B, T, 11]  (x,y,z,t + kinematic features)
        mask:   [B, T]      (True for valid, False for padding)
    Output:
        memory: [B, T, D]   encoded features
        coords: [B, T, 3]   original xyz coordinates
        mean:   [B, 1, 3]   (currently zeros, kept for API compatibility)
        scale:  [B, 1, 1]   (currently ones, kept for API compatibility)
    """

    def __init__(self, d_model: int = 128, nhead: int = 4, num_layers: int = 3):
        super().__init__()
        self.input_proj = nn.Linear(11, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, traces: torch.Tensor, mask: Optional[torch.Tensor] = None):
        B, N, _ = traces.shape
        coords = traces[..., :3].contiguous()

        # For now we do not normalize coordinates (keep mean=0, scale=1),
        # but keep the return values for compatibility with the decoder.
        mean = torch.zeros(B, 1, 3, device=traces.device, dtype=traces.dtype)
        scale = torch.ones(B, 1, 1, device=traces.device, dtype=traces.dtype)

        x = self.input_proj(traces)
        x = self.pos_encoder(x)

        src_key_padding_mask = ~mask if mask is not None else None
        memory = self.transformer(x, src_key_padding_mask=src_key_padding_mask)

        return memory, coords, mean, scale



class GridEncoderCNN(nn.Module):
    """
    Encode rasterized trace grid (Walk2Map-style inverse-distance map).

    Input:
        grid_maps: [B, 1, H, W] (values in [0,1])
    Output:
        grid_context: [B, D] (global 2D context embedding)
    """

    def __init__(self, d_model: int = 128, in_channels: int = 1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),  # 64 -> 32
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),          # 32 -> 16
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),         # 16 -> 8
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.proj = nn.Linear(128, d_model)

    def forward(self, grid_maps: torch.Tensor) -> torch.Tensor:
        x = self.conv(grid_maps)       # [B, 128, H', W']
        x = x.mean(dim=(2, 3))         # Global Average Pooling -> [B, 128]
        x = self.proj(x)               # [B, d_model]
        return x



class MLP(nn.Module):
    """Simple MLP used for bounding box heads."""

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



class ColliderDecoder(nn.Module):
    """
    DETR-style transformer decoder which:

    - Uses a set of learnable queries to predict colliders
    - Attends over trace-encoded features
    - Optionally conditions on a grid-level context vector
    """

    def __init__(self, d_model: int = 128, nhead: int = 4, num_layers: int = 3, num_queries: int = 30):
        super().__init__()
        self.num_queries = num_queries
        self.query_embed = nn.Embedding(num_queries, d_model)

        # Global context extracted from trace features
        self.context_query = nn.Parameter(torch.randn(1, 1, d_model))
        self.context_attn = nn.MultiheadAttention(d_model, num_heads=nhead, batch_first=True)

        # Simple 3D spatial statistics of the trace
        self.spatial_proj = nn.Sequential(
            nn.Linear(6, d_model),  # [mean(3) + std(3)]
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Heads
        self.center_delta_head = MLP(d_model, d_model, 3, 2)
        self.size_head = MLP(d_model, d_model, 3, 2)
        self.class_head = nn.Linear(d_model, 3)

        # Extra projections for anchor computation
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.scale = d_model ** 0.5

        # FiLM-style conditioning on global context (trace + grid)
        self.gamma_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model)
        )
        self.beta_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model)
        )

    def forward(
            self,
            memory: torch.Tensor,
            coords: torch.Tensor,
            mean: torch.Tensor,
            scale: torch.Tensor,
            memory_mask: Optional[torch.Tensor] = None,
            grid_context: Optional[torch.Tensor] = None
    ):
        """
        Args:
            memory:       [B, T, D] trace-encoded features
            coords:       [B, T, 3] original xyz coordinates
            mean:         [B, 1, 3] (currently zeros)
            scale:        [B, 1, 1] (currently ones)
            memory_mask:  [B, T] True = valid, False = padding
            grid_context: [B, D] optional global context from grid encoder

        Returns:
            boxes:   [B, Q, 6]  (cx, cy, cz, sx, sy, sz)
            classes: [B, Q, 3]  class logits (BLOCK / LOW / MID)
        """
        B, N, D = memory.shape

        # Learnable base queries
        base_queries = self.query_embed.weight.unsqueeze(0)  # [1, Q, D] -> [B, Q, D] later

        # Global context from trace features
        context_query = self.context_query.expand(B, -1, -1)
        key_padding_mask = ~memory_mask if memory_mask is not None else None
        global_context, _ = self.context_attn(
            context_query, memory, memory,
            key_padding_mask=key_padding_mask
        )  # [B, 1, D]

        # Compute simple spatial stats on coordinates
        if memory_mask is not None:
            valid_coords = coords * memory_mask.unsqueeze(-1)
            denom = memory_mask.sum(dim=1, keepdim=True).clamp_min(1).unsqueeze(-1)
            coord_mean = valid_coords.sum(dim=1, keepdim=True) / denom
            centered = (coords - coord_mean) * memory_mask.unsqueeze(-1)
            coord_var = (centered ** 2).sum(dim=1, keepdim=True) / denom
            coord_std = torch.sqrt(coord_var + 1e-6)
        else:
            coord_mean = coords.mean(dim=1, keepdim=True)
            coord_std = coords.std(dim=1, keepdim=True)

        spatial_stats = torch.cat([coord_mean, coord_std], dim=-1)  # [B, 1, 6]
        spatial_feat = self.spatial_proj(spatial_stats)             # [B, 1, D]

        # Combine trace-global, spatial, and grid context
        combined_context = global_context + spatial_feat  # [B, 1, D]
        if grid_context is not None:
            combined_context = combined_context + grid_context.unsqueeze(1)

        # FiLM conditioning of query embeddings
        gamma = self.gamma_mlp(combined_context)  # [B, 1, D]
        beta = self.beta_mlp(combined_context)   # [B, 1, D]
        queries = base_queries * (1.0 + gamma) + beta  # [B, Q, D]

        # Transformer decoder over trace memory
        mem_pad_mask = ~memory_mask if memory_mask is not None else None
        decoded = self.transformer(
            queries,
            memory,
            memory_key_padding_mask=mem_pad_mask
        )  # [B, Q, D]

        # Attention back to coordinates to get an anchor position per query
        q = self.q_proj(decoded)  # [B, Q, D]
        k = self.k_proj(memory)   # [B, T, D]
        attn_scores = torch.einsum('bqd,bnd->bqn', q, k) / self.scale  # [B, Q, T]

        if memory_mask is not None:
            pad = ~memory_mask
            attn_scores = attn_scores.masked_fill(pad.unsqueeze(1), float('-inf'))

        attn_weights = torch.softmax(attn_scores, dim=-1)  # [B, Q, T]

        # Normalize coordinates with mean/scale before computing anchor
        norm_coords = (coords - mean) / scale
        anchor_pos = torch.einsum('bqn,bnd->bqd', attn_weights, norm_coords)  # [B, Q, 3]

        # Box regression heads
        delta_center = self.center_delta_head(decoded)       # [B, Q, 3]
        size_raw = self.size_head(decoded)                  # [B, Q, 3]
        size_norm = F.softplus(size_raw) + 1e-4             # ensure positive sizes

        center = (anchor_pos + delta_center) * scale + mean # [B, Q, 3]
        size = size_norm * scale                            # [B, Q, 3]

        boxes = torch.cat([center, size], dim=-1)           # [B, Q, 6]
        classes = self.class_head(decoded)                  # [B, Q, 3]
        return boxes, classes


# Full model: Trace + Grid -> Colliders
class TraceToColliderTransformer(nn.Module):
    """
    Full Room-SLAM model (Transformer-based):

    - Encodes the 3D trace sequence with a Transformer
    - Encodes the Walk2Map-style 2D raster grid with a CNN
    - Fuses both through a transformer decoder to predict 3D colliders
    """

    def __init__(self,
                 d_model: int = 128,
                 nhead: int = 4,
                 num_encoder_layers: int = 3,
                 num_decoder_layers: int = 3,
                 num_queries: int = 30):
        super().__init__()

        self.encoder = TransformerTraceEncoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_encoder_layers
        )

        self.grid_encoder = GridEncoderCNN(d_model=d_model, in_channels=1)

        self.decoder = ColliderDecoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_decoder_layers,
            num_queries=num_queries
        )

    def forward(
        self,
        traces: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        grid_maps: Optional[torch.Tensor] = None
    ):
        """
        Args:
            traces:    [B, T, 11] trace features from dataloader
            mask:      [B, T] True = valid, False = padding
            grid_maps: [B, 1, H, W] Walk2Map-style inverse-distance maps

        Returns:
            dict with:
              - 'pred_boxes':   [B, Q, 6]
              - 'pred_classes': [B, Q, 3]
        """
        # Encode traces
        memory, coords, mean, scale = self.encoder(traces, mask)

        # Encode grid maps (2D context)
        if grid_maps is not None:
            grid_context = self.grid_encoder(grid_maps)  # [B, D]
        else:
            grid_context = None

        # Decode colliders from combined context
        boxes, classes = self.decoder(
            memory, coords, mean, scale,
            memory_mask=mask,
            grid_context=grid_context
        )
        return {
            'pred_boxes': boxes,
            'pred_classes': classes
        }


def build_model(
        num_queries: int = 80,
        d_model: int = 256,
        nhead: int = 8,
        enc_layers: int = 6,
        dec_layers: int = 6,
):
    """
    Build the transformer-only model.

    Note: LSTM/Baseline-collider options are removed for simplicity.
    """
    model = TraceToColliderTransformer(
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=enc_layers,
        num_decoder_layers=dec_layers,
        num_queries=num_queries,
    )
    print(f"[build_model] Transformer: d_model={d_model}, heads={nhead}, "
          f"enc/dec={enc_layers}/{dec_layers}, queries={num_queries}")
    return model


def count_parameters(model):
    """Count the number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device("cpu")
        print("CUDA not available, using CPU")

    print("\nTesting transformer model...")
    model = build_model(num_queries=30, d_model=128, nhead=4, enc_layers=3, dec_layers=3).to(device)

    num_params = count_parameters(model)
    print(f"Total trainable parameters: {num_params:,}")

    print("\nTesting forward pass...")
    batch_size = 2
    seq_len = 500
    H = W = 64

    traces = torch.randn(batch_size, seq_len, 11, device=device)
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
    grid_maps = torch.randn(batch_size, 1, H, W, device=device)

    output = model(traces, mask, grid_maps=grid_maps)

    print("Output shapes:")
    print(f"  Boxes:   {output['pred_boxes'].shape}")
    print(f"  Classes: {output['pred_classes'].shape}")

    print("\nTesting trace dependency (different directions)...")
    trace1 = torch.zeros(1, seq_len, 11, device=device)
    trace1[0, :, 0] = torch.linspace(-2, 2, seq_len, device=device)  # horizontal

    trace2 = torch.zeros(1, seq_len, 11, device=device)
    trace2[0, :, 1] = torch.linspace(-2, 2, seq_len, device=device)  # vertical

    mask1 = torch.ones(1, seq_len, dtype=torch.bool, device=device)
    grid1 = torch.randn(1, 1, H, W, device=device)

    with torch.no_grad():
        out1 = model(trace1, mask1, grid_maps=grid1)
        out2 = model(trace2, mask1, grid_maps=grid1)

    diff = (out1['pred_boxes'] - out2['pred_boxes']).abs().mean().item()
    print(f"  Horizontal vs Vertical difference: {diff:.4f}")

    if diff > 0.5:
        print("PASS: Model produces different outputs for different traces!")
    else:
        print("Predictions still similar, may need more training / stronger encoder")

    print("\nModel test passed!")