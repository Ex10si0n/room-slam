import math
import torch
import torch.nn as nn
from typing import Optional

class LSTMTraceEncoder(nn.Module):
    """
    Encode trace sequences with a BiLSTM.
    Input features are expected to be [x,y,z,t, vx,vy,vz, ax,ay,az, speed] (11-D).
    We keep the same normalization outputs (coords mean/scale) for relative decoding.
    """

    def __init__(self, input_dim: int = 11, d_model: int = 128, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model // 2,   # BiLSTM -> output dim = d_model
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=True
        )
        self.out_proj = nn.Linear(d_model, d_model)  # optional stabilization

    def forward(self, traces: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        Args:
            traces: [B, N, 11]
            mask:   [B, N] boolean; True means valid token

        Returns:
            memory: [B, N, D] BiLSTM features
            coords: [B, N, 3] raw xyz
            mean:   [B, 1, 3] per-batch mean over valid coords
            scale:  [B, 1, 1] per-batch RMS scale (x,z)
        """
        B, N, _ = traces.shape
        coords = traces[..., :3].contiguous()

        valid = mask if mask is not None else torch.ones((B, N), dtype=torch.bool, device=traces.device)
        denom = valid.sum(dim=1, keepdim=True).clamp_min(1).unsqueeze(-1)  # [B,1,1]
        mean = (coords * valid.unsqueeze(-1)).sum(dim=1, keepdim=True) / denom
        centered = (coords - mean) * valid.unsqueeze(-1)
        rms = torch.sqrt((centered[..., [0, 2]] ** 2).sum(dim=(1, 2), keepdim=True) / denom[..., :1]).clamp_min(1e-3)
        scale = rms

        x = self.input_proj(traces)  # [B,N,D]
        # LSTM can naturally ignore padded zeros; providing mask is optional
        memory, _ = self.lstm(x)     # [B,N,D]
        memory = self.out_proj(memory)

        return memory, coords, mean, scale


class SimpleQueryDecoder(nn.Module):
    """
    Query-based set decoder without a Transformer.
    It uses learnable queries and dot-product attention over the memory to
    get both anchor positions (via coords) and query features (via memory values).
    """

    def __init__(self, d_model: int = 128, num_queries: int = 30):
        super().__init__()
        self.num_queries = num_queries
        self.query_embed = nn.Embedding(num_queries, d_model)

        # Linear projections for attention
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.scale = d_model ** 0.5

        # Heads (same as Transformer decoder variant)
        self.center_delta_head = MLP(d_model, d_model, 3, 2)
        self.size_head = MLP(d_model, d_model, 3, 2)
        self.class_head = nn.Linear(d_model, 4)

        # Optional FiLM from global memory summary to modulate decoded features
        self.gamma_mlp = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
        self.beta_mlp  = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, d_model))

        # Learnable attention temperature
        self.inv_temp = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        memory: torch.Tensor,               # [B,N,D]
        coords: torch.Tensor,               # [B,N,3]
        mean: torch.Tensor,                 # [B,1,3]
        scale: torch.Tensor,                # [B,1,1]
        memory_mask: Optional[torch.Tensor] = None  # [B,N] True for valid
    ):
        B, N, D = memory.shape
        queries = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)   # [B,Q,D]

        # Global summary for FiLM
        if memory_mask is not None:
            denom = memory_mask.sum(dim=1, keepdim=True).clamp_min(1).unsqueeze(-1)
            global_feat = (memory * memory_mask.unsqueeze(-1)).sum(dim=1, keepdim=True) / denom  # [B,1,D]
        else:
            global_feat = memory.mean(dim=1, keepdim=True)

        gamma = self.gamma_mlp(global_feat)   # [B,1,D]
        beta  = self.beta_mlp(global_feat)    # [B,1,D]

        # Attention over memory to get query-aligned features
        q = self.q_proj(queries)                      # [B,Q,D]
        k = self.k_proj(memory)                       # [B,N,D]
        v = self.v_proj(memory)                       # [B,N,D]
        scores = torch.einsum('bqd,bnd->bqn', q, k) * self.inv_temp / self.scale  # [B,Q,N]

        if memory_mask is not None:
            pad = ~memory_mask  # True where padded
            scores = scores.masked_fill(pad.unsqueeze(1), float('-inf'))

        attn = torch.softmax(scores, dim=-1)          # [B,Q,N]
        qfeat = torch.einsum('bqn,bnd->bqd', attn, v) # [B,Q,D]

        # Apply FiLM modulation
        decoded = qfeat * (1.0 + gamma) + beta        # [B,Q,D]

        # Anchor from normalized coords
        norm_coords = (coords - mean) / scale         # [B,N,3]
        anchor_pos = torch.einsum('bqn,bnd->bqd', attn, norm_coords)  # [B,Q,3]

        delta_center = self.center_delta_head(decoded)     # [B,Q,3]
        size_raw     = self.size_head(decoded)             # [B,Q,3]
        size_norm    = torch.nn.functional.softplus(size_raw) + 1e-4

        center = (anchor_pos + delta_center) * scale + mean
        size   = size_norm * scale

        boxes   = torch.cat([center, size], dim=-1)        # [B,Q,6]
        classes = self.class_head(decoded)                  # [B,Q,4]
        return boxes, classes


class TraceToColliderLSTM(nn.Module):
    """
    LSTM encoder + simple query decoder.
    Produces {'pred_boxes': [B,Q,6], 'pred_classes': [B,Q,4]}.
    """

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

        self.input_proj = nn.Linear(11, d_model)

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
        coords = traces[..., :3].contiguous()  # [B,N,3] raw xyz
        valid = mask if mask is not None else torch.ones(traces.size()[:2], dtype=torch.bool, device=traces.device)

        # mean over valid points
        denom = valid.sum(dim=1, keepdim=True).clamp_min(1).unsqueeze(-1)  # [B,1,1]
        mean = (coords * valid.unsqueeze(-1)).sum(dim=1, keepdim=True) / denom  # [B,1,3]

        # robust scale: RMS of centered coords in x,z (y可选)
        centered = (coords - mean) * valid.unsqueeze(-1)
        rms = torch.sqrt((centered[..., [0, 2]] ** 2).sum(dim=(1, 2), keepdim=True) / denom[..., :1]).clamp_min(1e-3)
        scale = rms  # scalar per batch (use x,z energy)

        x = self.input_proj(traces)  # [B,N,D]
        x = self.pos_encoding(x)
        attn_mask = ~mask if mask is not None else None
        encoded = self.transformer(x, src_key_padding_mask=attn_mask)

        return encoded, coords, mean, scale


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
        self.size_head = MLP(d_model, d_model, 3, 2)  # raw logits -> softplus
        self.class_head = nn.Linear(d_model, 4)  # BLOCK/LOW/MID/HIGH

        # Lightweight attention projections for anchor computation
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.scale = d_model ** 0.5  # for dot-product attention scaling

    def forward(
            self,
            memory: torch.Tensor,  # [B, N, D] encoded features
            coords: torch.Tensor,  # [B, N, 3] raw (x,y,z) of traces
            mean: torch.Tensor,  # [B, 1, 3] mean of valid coords
            scale: torch.Tensor,  # [B, 1, 1] scale of
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
        q = self.q_proj(decoded)  # [B, Q, D]
        k = self.k_proj(memory)  # [B, N, D]
        attn_scores = torch.einsum('bqd,bnd->bqn', q, k) / self.scale  # [B, Q, N]

        if memory_mask is not None:
            # Mask out padded positions: set them to a very negative value
            # memory_mask: True for valid -> we want False for padded, so invert:
            pad = ~memory_mask  # True where padded
            attn_scores = attn_scores.masked_fill(pad.unsqueeze(1), float('-inf'))

        # normalize coords to canonical space
        norm_coords = (coords - mean) / scale  # [B,N,3]

        attn_weights = torch.softmax(attn_scores, dim=-1)  # [B, Q, N]

        # Anchor position is the attention-weighted average of raw coords
        anchor_pos = torch.einsum('bqn,bnd->bqd', attn_weights, norm_coords)  # [B,Q,3]

        # relative predictions in normalized space
        delta_center = self.center_delta_head(decoded)  # [B,Q,3]
        size_raw = self.size_head(decoded)  # [B,Q,3]
        size_norm = torch.nn.functional.softplus(size_raw) + 1e-4

        # denormalize back to absolute
        center = anchor_pos + delta_center  # normalized
        center = center * scale + mean  # absolute
        size = size_norm * scale  # absolute (isotropic scale;也可只乘x,z)

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
        memory, coords, mean, scale = self.encoder(traces, mask)  # memory:[B,N,D], coords:[B,N,3]

        # Decode colliders with anchor-relative prediction
        boxes, classes = self.decoder(memory, coords, mean, scale, mask)

        return {
            'pred_boxes': boxes,  # [B, Q, 6] absolute boxes
            'pred_classes': classes  # [B, Q, 4] class logits
        }


def build_model(
    num_queries: int = 80,
    d_model: int = 256,
    model_type: str = "transformer",   # 'transformer' or 'lstm'
    nhead: int = 8,
    enc_layers: int = 6,
    dec_layers: int = 6,
    dim_feedforward: int = 2048,
    dropout: float = 0.1,
    lstm_layers: int = 2
):
    """
    Build model by type. Both variants output the same dict interface.
    """
    model_type = model_type.lower()
    if model_type == "transformer":
        model = TraceToColliderTransformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=enc_layers,
            num_decoder_layers=dec_layers,
            num_queries=num_queries
        )
        print(f"[build_model] Using Transformer: d_model={d_model}, heads={nhead}, enc/dec={enc_layers}/{dec_layers}, queries={num_queries}")
        return model

    elif model_type == "lstm":
        model = TraceToColliderLSTM(
            d_model=d_model,
            num_queries=num_queries,
            lstm_layers=lstm_layers,
            dropout=dropout
        )
        print(f"[build_model] Using LSTM: d_model={d_model}, layers={lstm_layers}, queries={num_queries}")
        return model

    else:
        raise ValueError(f"Unknown model_type: {model_type}. Use 'transformer' or 'lstm'.")


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
