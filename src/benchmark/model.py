import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class LSTMTraceEncoder(nn.Module):
    """Encode trace sequences with a BiLSTM."""

    def __init__(self, input_dim: int = 11, d_model: int = 128, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model // 2,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=True
        )
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, traces: torch.Tensor, mask: Optional[torch.Tensor] = None):
        B, N, _ = traces.shape
        coords = traces[..., :3].contiguous()

        valid = mask if mask is not None else torch.ones((B, N), dtype=torch.bool, device=traces.device)
        denom = valid.sum(dim=1, keepdim=True).clamp_min(1).unsqueeze(-1)

        mean = (coords * valid.unsqueeze(-1)).sum(dim=1, keepdim=True) / denom
        centered = (coords - mean) * valid.unsqueeze(-1)
        rms = torch.sqrt((centered[..., [0, 2]] ** 2).sum(dim=(1, 2), keepdim=True) / denom[..., :1]).clamp_min(1e-3)
        scale = rms

        x = self.input_proj(traces)
        memory, _ = self.lstm(x)
        memory = self.out_proj(memory)

        return memory, coords, mean, scale


class SimpleQueryDecoder(nn.Module):

    def __init__(self, d_model: int = 128, num_queries: int = 30):
        super().__init__()
        self.num_queries = num_queries
        self.query_embed = nn.Embedding(num_queries, d_model)

        self.context_query = nn.Parameter(torch.randn(1, 1, d_model))
        self.context_attn = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)

        self.spatial_proj = nn.Sequential(
            nn.Linear(6, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

        # Attention
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.scale = d_model ** 0.5

        # Heads
        self.center_delta_head = MLP(d_model, d_model, 3, 2)
        self.size_head = MLP(d_model, d_model, 3, 2)
        self.class_head = nn.Linear(d_model, 3)

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

        self.inv_temp = nn.Parameter(torch.tensor(1.0))

    def forward(
            self,
            memory: torch.Tensor,
            coords: torch.Tensor,
            mean: torch.Tensor,
            scale: torch.Tensor,
            memory_mask: Optional[torch.Tensor] = None
    ):
        B, N, D = memory.shape

        base_queries = self.query_embed.weight.unsqueeze(0)

        context_query = self.context_query.expand(B, -1, -1)
        key_padding_mask = ~memory_mask if memory_mask is not None else None
        global_context, _ = self.context_attn(
            context_query, memory, memory,
            key_padding_mask=key_padding_mask
        )  # [B, 1, D]

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
        spatial_feat = self.spatial_proj(spatial_stats)  # [B, 1, D]

        combined_context = global_context + spatial_feat

        gamma = self.gamma_mlp(combined_context)
        beta = self.beta_mlp(combined_context)
        queries = base_queries * (1.0 + gamma) + beta  # [B, Q, D]

        # Attention over memory
        q = self.q_proj(queries)
        k = self.k_proj(memory)
        v = self.v_proj(memory)
        scores = torch.einsum('bqd,bnd->bqn', q, k) * self.inv_temp / self.scale

        if memory_mask is not None:
            pad = ~memory_mask
            scores = scores.masked_fill(pad.unsqueeze(1), float('-inf'))

        attn = torch.softmax(scores, dim=-1)
        qfeat = torch.einsum('bqn,bnd->bqd', attn, v)

        # Apply FiLM again
        decoded = qfeat * (1.0 + gamma) + beta

        # Anchor
        norm_coords = (coords - mean) / scale
        anchor_pos = torch.einsum('bqn,bnd->bqd', attn, norm_coords)

        delta_center = self.center_delta_head(decoded)
        size_raw = self.size_head(decoded)
        size_norm = F.softplus(size_raw) + 1e-4

        center = (anchor_pos + delta_center) * scale + mean
        size = size_norm * scale

        boxes = torch.cat([center, size], dim=-1)
        classes = self.class_head(decoded)
        return boxes, classes


class TraceToColliderLSTM(nn.Module):
    """LSTM encoder + improved query decoder."""

    def __init__(self, d_model: int = 128, num_queries: int = 30, lstm_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.encoder = LSTMTraceEncoder(input_dim=11, d_model=d_model, num_layers=lstm_layers, dropout=dropout)
        self.decoder = SimpleQueryDecoder(d_model=d_model, num_queries=num_queries)

    def forward(self, traces: torch.Tensor, mask: Optional[torch.Tensor] = None):
        memory, coords, mean, scale = self.encoder(traces, mask)
        boxes, classes = self.decoder(memory, coords, mean, scale, mask)
        return {'pred_boxes': boxes, 'pred_classes': classes}


class PositionalEncoding(nn.Module):
    """3D + Temporal positional encoding with dynamic length support"""

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
        seq_len = x.size(1)
        if seq_len > self.max_len:
            self._extend_pe(seq_len, x.device)
        return x + self.pe[:seq_len, :].unsqueeze(0)

    def _extend_pe(self, new_len: int, device):
        pe = torch.zeros(new_len, self.d_model, device=device)
        position = torch.arange(0, new_len, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2, device=device).float() *
                             (-math.log(10000.0) / self.d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)


class TransformerTraceEncoder(nn.Module):
    """Transformer encoder for trace sequences"""

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

        valid = mask if mask is not None else torch.ones((B, N), dtype=torch.bool, device=traces.device)
        denom = valid.sum(dim=1, keepdim=True).clamp_min(1).unsqueeze(-1)

        mean = (coords * valid.unsqueeze(-1)).sum(dim=1, keepdim=True) / denom
        centered = (coords - mean) * valid.unsqueeze(-1)
        rms = torch.sqrt((centered[..., [0, 2]] ** 2).sum(dim=(1, 2), keepdim=True) / denom[..., :1]).clamp_min(1e-3)
        scale = rms

        x = self.input_proj(traces)
        x = self.pos_encoder(x)

        src_key_padding_mask = ~mask if mask is not None else None
        memory = self.transformer(x, src_key_padding_mask=src_key_padding_mask)

        return memory, coords, mean, scale


class ColliderDecoder(nn.Module):

    def __init__(self, d_model: int = 128, nhead: int = 4, num_layers: int = 3, num_queries: int = 30):
        super().__init__()
        self.num_queries = num_queries
        self.query_embed = nn.Embedding(num_queries, d_model)

        self.context_query = nn.Parameter(torch.randn(1, 1, d_model))
        self.context_attn = nn.MultiheadAttention(d_model, num_heads=nhead, batch_first=True)

        self.spatial_proj = nn.Sequential(
            nn.Linear(6, d_model),
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

        self.center_delta_head = MLP(d_model, d_model, 3, 2)
        self.size_head = MLP(d_model, d_model, 3, 2)
        self.class_head = nn.Linear(d_model, 3)

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.scale = d_model ** 0.5

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
            memory_mask: Optional[torch.Tensor] = None
    ):
        B, N, D = memory.shape

        base_queries = self.query_embed.weight.unsqueeze(0)

        context_query = self.context_query.expand(B, -1, -1)
        key_padding_mask = ~memory_mask if memory_mask is not None else None
        global_context, _ = self.context_attn(
            context_query, memory, memory,
            key_padding_mask=key_padding_mask
        )

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

        spatial_stats = torch.cat([coord_mean, coord_std], dim=-1)
        spatial_feat = self.spatial_proj(spatial_stats)
        combined_context = global_context + spatial_feat

        gamma = self.gamma_mlp(combined_context)
        beta = self.beta_mlp(combined_context)
        queries = base_queries * (1.0 + gamma) + beta

        mem_pad_mask = ~memory_mask if memory_mask is not None else None
        decoded = self.transformer(
            queries,
            memory,
            memory_key_padding_mask=mem_pad_mask
        )

        q = self.q_proj(decoded)
        k = self.k_proj(memory)
        attn_scores = torch.einsum('bqd,bnd->bqn', q, k) / self.scale

        if memory_mask is not None:
            pad = ~memory_mask
            attn_scores = attn_scores.masked_fill(pad.unsqueeze(1), float('-inf'))

        norm_coords = (coords - mean) / scale
        attn_weights = torch.softmax(attn_scores, dim=-1)
        anchor_pos = torch.einsum('bqn,bnd->bqd', attn_weights, norm_coords)

        delta_center = self.center_delta_head(decoded)
        size_raw = self.size_head(decoded)
        size_norm = F.softplus(size_raw) + 1e-4

        center = anchor_pos + delta_center
        center = center * scale + mean
        size = size_norm * scale

        boxes = torch.cat([center, size], dim=-1)
        classes = self.class_head(decoded)
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
    """Trace -> Colliders with improved decoder"""

    def __init__(self, d_model: int = 128, nhead: int = 4,
                 num_encoder_layers: int = 3, num_decoder_layers: int = 3,
                 num_queries: int = 30):
        super().__init__()

        self.encoder = TransformerTraceEncoder(
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
        memory, coords, mean, scale = self.encoder(traces, mask)
        boxes, classes = self.decoder(memory, coords, mean, scale, mask)
        return {
            'pred_boxes': boxes,
            'pred_classes': classes
        }


def build_model(
        num_queries: int = 80,
        d_model: int = 256,
        model_type: str = "transformer",
        nhead: int = 8,
        enc_layers: int = 6,
        dec_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        lstm_layers: int = 2
):
    """Build model by type."""
    model_type = model_type.lower()
    if model_type == "transformer":
        model = TraceToColliderTransformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=enc_layers,
            num_decoder_layers=dec_layers,
            num_queries=num_queries
        )
        print(f"[build_model] Transformer: d_model={d_model}, heads={nhead}, "
              f"enc/dec={enc_layers}/{dec_layers}, queries={num_queries}")
        return model

    elif model_type == "lstm":
        model = TraceToColliderLSTM(
            d_model=d_model,
            num_queries=num_queries,
            lstm_layers=lstm_layers,
            dropout=dropout
        )
        print(f"[build_model] LSTM: d_model={d_model}, layers={lstm_layers}, queries={num_queries}")
        return model

    else:
        raise ValueError(f"Unknown model_type: {model_type}. Use 'transformer' or 'lstm'.")


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device("cpu")
        print("CUDA not available, using CPU")

    print("\nTesting FIXED model...")
    model = build_model(num_queries=30, d_model=128, model_type='transformer').to(device)

    num_params = count_parameters(model)
    print(f"Total trainable parameters: {num_params:,}")

    print("\nTesting forward pass...")
    batch_size = 2
    seq_len = 500
    traces = torch.randn(batch_size, seq_len, 11, device=device)
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)

    output = model(traces, mask)

    print("Output shapes:")
    print(f"  Boxes: {output['pred_boxes'].shape}")
    print(f"  Classes: {output['pred_classes'].shape}")

    print("\nTesting trace dependency...")
    # Different traces with same mean
    trace1 = torch.zeros(1, seq_len, 11, device=device)
    trace1[0, :, 0] = torch.linspace(-2, 2, seq_len, device=device)  # horizontal

    trace2 = torch.zeros(1, seq_len, 11, device=device)
    trace2[0, :, 1] = torch.linspace(-2, 2, seq_len, device=device)  # vertical

    mask1 = torch.ones(1, seq_len, dtype=torch.bool, device=device)

    with torch.no_grad():
        out1 = model(trace1, mask1)
        out2 = model(trace2, mask1)

    diff = (out1['pred_boxes'] - out2['pred_boxes']).abs().mean().item()
    print(f"  Horizontal vs Vertical difference: {diff:.4f}")

    if diff > 0.5:
        print(f"PASS: Model produces different outputs for different traces!")
    else:
        print(f"Predictions still similar, may need more training")

    print("\nModel test passed!")