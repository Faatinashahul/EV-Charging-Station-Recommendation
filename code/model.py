"""
model.py
--------
The complete STGAT architecture.

Components (in forward-pass order):
    1. FeatureFusion   — Conv2d fuses occupancy + price into one stream per node
    2. GATLayer        — Sparse multi-head graph attention (fixed: nn.ParameterDict)
    3. SpatialEncoder  — Two GATLayers with alpha-weighted residual connections
    4. TPADecoder      — 2-layer LSTM + Temporal Pattern Attention → scalar per node
    5. STGAT           — Assembles all of the above; final Sigmoid clamps output to (0,1)

Public API:
    build_model(cfg, adj_sparse) → STGAT
    
    The only function other files need. Returns a fully initialized model
    on the correct device, with the adjacency matrix registered as a buffer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config


# ---------------------------------------------------------------------------
# 1. FeatureFusion
# ---------------------------------------------------------------------------

class FeatureFusion(nn.Module):
    """
    Fuses occupancy and price into a single feature sequence per node
    using a 2D convolution.

    Input:
        occ : (B, N, S)   occupancy sequences, node-first
        prc : (B, N, S)   price sequences, node-first

    Internal reshape:
        Stack along feature dim → (B*N, 1, S, 2)
        Conv2d kernel (conv_kernel, 2) collapses the feature dimension
        and slightly compresses the sequence dimension.

    Output:
        (B, N, S')  where S' = S - conv_kernel + 1
        With default S=12, conv_kernel=2 → S'=11.

    Why Conv2d and not just concatenate?
        Concatenation would pass occ and price as separate channels to the GAT,
        which would need to learn their interaction implicitly across many layers.
        The Conv2d learns a *joint* local pattern — e.g. "high price + rising occ"
        — as a single compressed feature, giving the GAT a richer starting signal.

    Why kernel size (conv_kernel, 2)?
        The '2' dimension collapses the two input features (occ, price) → 1.
        The 'conv_kernel' dimension slides along time, creating a local
        temporal summary. The output channel is 1, so no feature explosion.
    """

    def __init__(self, conv_kernel: int, n_features: int = 2):
        super().__init__()
        # in_channels=1  : single channel wrapping the (S, F) feature map
        # out_channels=1 : single fused channel
        # kernel_size=(conv_kernel, n_features) : local time × all features
        self.conv = nn.Conv2d(
            in_channels  = 1,
            out_channels = 1,
            kernel_size  = (conv_kernel, n_features),
        )
        self.act = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, occ: torch.Tensor, prc: torch.Tensor) -> torch.Tensor:
        B, N, S = occ.shape

        # Stack features: (B, N, S, 2)
        x = torch.stack([occ, prc], dim=-1)

        # Reshape for Conv2d: (B*N, 1, S, 2)
        x = x.view(B * N, 1, S, 2)

        # Convolve: (B*N, 1, S', 1)  where S' = S - conv_kernel + 1
        x = self.conv(x)
        x = self.act(x)

        # Squeeze channels and restore batch+node: (B, N, S')
        x = x.squeeze(-1).squeeze(1)   # (B*N, S')
        x = x.view(B, N, -1)           # (B, N, S')

        return x


# ---------------------------------------------------------------------------
# 2. GATLayer
# ---------------------------------------------------------------------------

class GATLayer(nn.Module):
    """
    Sparse multi-head Graph Attention Network layer.

    Key fix vs. original PAG:
        Parameters stored in nn.ParameterDict instead of plain dict.
        PyTorch's optimizer scans nn.ParameterDict during model.parameters(),
        so attention weights are now actually trained. In the original code
        they were initialized and then frozen for the entire training run.

    Attention mechanism (per head):
        1. Linear projection: h = x @ W           (B*N, S') → (B*N, out_dim)
        2. Compute edge scores: e_ij = a^T [h_i || h_j]
           Only computed for edges in the adjacency (sparse operation).
        3. Mask non-edges with -1e9, apply softmax → attention coefficients
        4. Aggregate: output[i] = sum_j( alpha_ij * h_j )

    Multi-head aggregation:
        Each head produces a scalar attention score per edge.
        Scores from all heads are stacked and linearly combined → one
        attention matrix. This is more expressive than averaging heads.

    Args:
        adj_sparse  : sparse COO adjacency tensor (N, N), registered as buffer
        input_dim   : feature dimension of input (= S' after FeatureFusion)
        out_dim     : output feature dimension (kept equal to input_dim here,
                      so residuals can be added without projection)
        n_heads     : number of attention heads
        dropout     : dropout on attention coefficients (0 = disabled)
        alpha       : LeakyReLU negative slope for attention scoring
    """

    def __init__(
        self,
        adj_sparse: torch.Tensor,
        input_dim:  int,
        out_dim:    int,
        n_heads:    int,
        dropout:    float,
        alpha:      float,
    ):
        super().__init__()

        self.n_heads   = n_heads
        self.input_dim = input_dim
        self.out_dim   = out_dim

        # ---- Attention parameters (properly registered) --------------------
        # W[h]  : (input_dim, out_dim)  — projection matrix per head
        # a[h]  : (1, 2*out_dim)        — scoring vector per head
        self.W = nn.ParameterDict({
            str(h): nn.Parameter(torch.empty(input_dim, out_dim))
            for h in range(n_heads)
        })
        self.a = nn.ParameterDict({
            str(h): nn.Parameter(torch.empty(1, 2 * out_dim))
            for h in range(n_heads)
        })

        # Xavier initialization — standard for attention projections
        for h in range(n_heads):
            nn.init.xavier_normal_(self.W[str(h)], gain=1.414)
            nn.init.xavier_normal_(self.a[str(h)], gain=1.414)

        # Linear combination of multi-head scores → single attention matrix
        self.head_combine = nn.Linear(n_heads, 1, bias=False)

        # ---- Regularization ------------------------------------------------
        self.leaky_relu = nn.LeakyReLU(negative_slope=alpha)
        self.dropout    = nn.Dropout(p=dropout)
        self.softmax    = nn.Softmax(dim=0)

        # ---- Graph structure (registered as buffers, move with .to(device)) -
        # Buffers are not parameters — optimizer ignores them, but they move
        # to GPU/MPS automatically when you call model.to(device).
        edges  = adj_sparse.coalesce().indices()   # (2, E) — source, dest indices
        values = adj_sparse.coalesce().values()    # (E,)   — edge weights (all 1.0)
        N      = adj_sparse.shape[0]

        self.register_buffer("edges",  edges)
        self.register_buffer("values", values.clone())
        self.N = N

        # Mask: -1e9 for non-edges, 0 for edges
        # Added to raw attention scores before softmax to zero out non-neighbors
        adj_dense = adj_sparse.to_dense()
        mask = torch.zeros_like(adj_dense)
        mask[adj_dense == 0] = -1e9
        self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, N, S')
        Returns:
            out : (B, N, S')
        Likely improvements in speed and accuracy (tho little)
        """
        B, N, S = x.shape

        head_scores = []

        for h in range(self.n_heads):
            # Project all nodes across all batches at once
            # x: (B, N, S) → (B, N, out_dim)
            h_feat = torch.matmul(x, self.W[str(h)])      # (B, N, out_dim)

            # Gather src and dst features for all edges, all batches
            # edges[0]: (E,) source indices, edges[1]: (E,) dest indices
            src = h_feat[:, self.edges[0], :]              # (B, E, out_dim)
            dst = h_feat[:, self.edges[1], :]              # (B, E, out_dim)

            # Concatenate and score: (B, E, 2*out_dim) → (B, E)
            edge_cat = torch.cat([src, dst], dim=-1)       # (B, E, 2*out_dim)
            # a[h]: (1, 2*out_dim) → squeeze to (2*out_dim,)
            a_vec    = self.a[str(h)].squeeze(0)           # (2*out_dim,)
            score    = (edge_cat * a_vec).sum(dim=-1)      # (B, E)
            score    = self.leaky_relu(score)              # (B, E)

            # Average across batch → (E,)
            head_scores.append(score.mean(dim=0))

        # Combine heads
        mt_scores  = torch.stack(head_scores, dim=1)       # (E, n_heads)
        comb_score = self.head_combine(mt_scores).squeeze(-1)  # (E,)

        # Build attention matrix — dense, MPS-safe
        scaled_values = self.values * comb_score
        attn_dense    = torch.zeros(self.N, self.N,
                                    device=x.device, dtype=scaled_values.dtype)
        attn_dense[self.edges[0], self.edges[1]] = scaled_values
        attn_dense    = attn_dense + self.mask
        attn_dense    = self.softmax(attn_dense)           # (N, N)

        out = torch.einsum('nm,bms->bns', attn_dense, x) #Againn

        return out


# ---------------------------------------------------------------------------
# 3. SpatialEncoder
# ---------------------------------------------------------------------------

class SpatialEncoder(nn.Module):
    """
    Two stacked GATLayers with alpha-weighted residual connections.

    Residual connections serve two purposes here:
        1. Prevent over-smoothing — without them, stacking GAT layers causes
           all node features to converge toward the graph mean (losing the
           individual zone signal).
        2. Gradient flow — skip connections give gradients a direct path back
           through the network, stabilizing training.

    Residual formula (per layer):
        output = (1 - alpha) * gat_output + alpha * layer_input

    alpha=0.5 means equal weighting of aggregated and original signal.
    Increase alpha if the model loses spatial diversity; decrease it if
    you want stronger neighborhood aggregation.

    Two layers gives a 2-hop receptive field — each node can "see" demand
    pressure from zones two edges away. Three+ layers risk over-smoothing
    and rarely help for graphs this sparse.
    """

    def __init__(
        self,
        adj_sparse:     torch.Tensor,
        seq_len:        int,          # S' (after FeatureFusion)
        n_heads:        int,
        dropout:        float,
        gat_alpha:      float,        # LeakyReLU slope inside GAT
        residual_alpha: float,        # residual skip weight
    ):
        super().__init__()

        self.residual_alpha = residual_alpha
        self.dropout        = nn.Dropout(p=dropout)
        self.act            = nn.LeakyReLU(negative_slope=0.2)

        # Both GAT layers keep the same feature dimension (seq_len → seq_len)
        # so residuals can be added without any projection layer
        self.gat1 = GATLayer(adj_sparse, seq_len, seq_len, n_heads, 0.0, gat_alpha)
        self.gat2 = GATLayer(adj_sparse, seq_len, seq_len, n_heads, 0.0, gat_alpha)

        # Linear transform applied after each GAT aggregation
        # Equivalent to GCN's self-weight matrix
        self.gcn1 = nn.Linear(seq_len, seq_len)
        self.gcn2 = nn.Linear(seq_len, seq_len)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x : (B, N, S')  fused feature sequences

        Returns:
            layer1_out : (B, N, S')  output of first GAT layer (used by LSTM)
            layer2_out : (B, N, S')  output of second GAT layer (used by LSTM)

        Both outputs are returned because TPADecoder stacks them as two
        features per timestep for the LSTM, giving the temporal module
        access to both 1-hop and 2-hop spatial context.
        """
        # ---- Layer 1 -------------------------------------------------------
        agg1     = self.gat1(x)                                      # (B, N, S')
        agg1     = self.dropout(self.act(self.gcn1(agg1)))
        layer1   = (1 - self.residual_alpha) * agg1 + \
                       self.residual_alpha  * x                      # (B, N, S')

        # ---- Layer 2 -------------------------------------------------------
        agg2     = self.gat2(layer1)                                  # (B, N, S')
        agg2     = self.dropout(self.act(self.gcn2(agg2)))
        layer2   = (1 - self.residual_alpha) * agg2 + \
                       self.residual_alpha  * layer1                  # (B, N, S')

        return layer1, layer2


# ---------------------------------------------------------------------------
# 4. TPADecoder
# ---------------------------------------------------------------------------

class TPADecoder(nn.Module):
    """
    Temporal Pattern Attention (TPA) decoder.

    Standard LSTM prediction uses only the final hidden state h_T.
    For EV charging demand, this misses strong periodic patterns —
    the demand shape from 8 hours ago is often highly predictive of
    the current forecast, but h_T has limited capacity to retain it.

    TPA solves this by:
        1. Running all LSTM hidden states through a learned projection
        2. Computing attention scores between each past state and h_T
        3. Forming a context vector v_T = attention-weighted sum of past states
        4. Concatenating v_T with h_T and projecting to a scalar prediction

    Input to LSTM:
        The two GAT layer outputs are stacked as two features per timestep:
        x[:, t, :] = [layer1_output_t, layer2_output_t]
        Shape: (B*N, S', 2) — each of the S' timesteps has 2 features.
        This gives the temporal module access to both 1-hop and 2-hop
        spatial context at each point in the sequence.

    Args:
        seq_len     : S' (internal sequence length after FeatureFusion)
        lstm_hidden : hidden state size (= 2 to match input feature count)
        lstm_layers : number of stacked LSTM layers
        tpa_k       : intermediate projection dimension in TPA attention
    """

    def __init__(
        self,
        seq_len:     int,
        lstm_hidden: int,
        lstm_layers: int,
        tpa_k:       int,
    ):
        super().__init__()

        self.seq_len = seq_len

        # LSTM — input size 2 (one feature per GAT layer)
        self.lstm = nn.LSTM(
            input_size  = 2,
            hidden_size = lstm_hidden,
            num_layers  = lstm_layers,
            batch_first = True,
        )

        # TPA projections
        # fc1: maps H_W (all hidden states except last) → key space
        #      H_W has shape (B*N, lstm_hidden, S'-1) after transpose
        #      fc1: (S'-1) → tpa_k
        self.fc1 = nn.Linear(seq_len - 1, tpa_k)

        # fc2: maps key space → query space matching lstm_hidden
        self.fc2 = nn.Linear(tpa_k, lstm_hidden)

        # fc3: maps [context_vector || h_T] → scalar prediction per node
        self.fc3 = nn.Linear(tpa_k + lstm_hidden, 1)

    def forward(
        self,
        layer1: torch.Tensor,   # (B, N, S')
        layer2: torch.Tensor,   # (B, N, S')
    ) -> torch.Tensor:
        """
        Returns:
            predictions : (B, N)  raw logits before sigmoid
        """
        B, N, S = layer1.shape

        # Stack GAT outputs as 2 features per node per timestep
        # (B, N, S', 2) → reshape to (B*N, S', 2) for LSTM
        x = torch.stack([layer1, layer2], dim=-1)   # (B, N, S', 2)
        x = x.view(B * N, S, 2)                     # (B*N, S', 2)

        # LSTM forward — all hidden states
        lstm_out, _ = self.lstm(x)                  # (B*N, S', lstm_hidden)

        # Split: final hidden state h_T and all previous h_1 ... h_{T-1}
        h_T = lstm_out[:, -1, :]                    # (B*N, lstm_hidden)
        H_W = lstm_out[:, :-1, :]                   # (B*N, S'-1, lstm_hidden)

        # TPA attention
        # H_W : (B*N, S'-1, lstm_hidden)
        # Transpose → (B*N, lstm_hidden, S'-1), then project along S'-1 → tpa_k
        H_W_t = H_W.transpose(1, 2)                 # (B*N, lstm_hidden, S'-1)
        H_c   = self.fc1(H_W_t)                     # (B*N, lstm_hidden, tpa_k)
        H_n   = self.fc2(H_c)                       # (B*N, lstm_hidden, lstm_hidden)

        # Attention scores: how much does each past hidden dim match h_T?
        # H_n^T : (B*N, lstm_hidden, lstm_hidden)
        # h_T_e : (B*N, lstm_hidden, 1)
        # result: (B*N, lstm_hidden, 1)
        h_T_e  = h_T.unsqueeze(2)                   # (B*N, lstm_hidden, 1)
        scores = torch.bmm(H_n.transpose(1, 2), h_T_e)  # (B*N, lstm_hidden, 1)
        scores = torch.sigmoid(scores)              # (B*N, lstm_hidden, 1)
        scores = scores.transpose(1, 2)             # (B*N, 1, lstm_hidden)

        # Context vector: scores (B*N, 1, lstm_hidden) @ H_c (B*N, lstm_hidden, tpa_k)
        #                 → (B*N, 1, tpa_k)
        v_T = torch.bmm(scores, H_c)               # (B*N, 1, tpa_k)
        v_T = v_T.squeeze(1)                        # (B*N, tpa_k)

        # Combine context and final hidden state → prediction
        h_T_final = h_T                              # (B*N, lstm_hidden)
        combined  = torch.cat([v_T, h_T_final], dim=1)  # (B*N, tpa_k + lstm_hidden)
        out = self.fc3(combined)                     # (B*N, 1)
        out = out.view(B, N)                         # (B, N)

        return out


# ---------------------------------------------------------------------------
# 5. STGAT — the full model
# ---------------------------------------------------------------------------

class STGAT(nn.Module):
    """
    Spatio-Temporal Graph Attention Network for EV charging demand prediction.

    Full forward pass:
        occ, prc [B, N, S]
            ↓ FeatureFusion (Conv2d)
        fused [B, N, S']
            ↓ SpatialEncoder (GATLayer × 2 + residuals)
        layer1, layer2 [B, N, S']
            ↓ TPADecoder (LSTM + TPA)
        logits [B, N]
            ↓ Sigmoid
        predictions [B, N]  ∈ (0, 1)

    The Sigmoid hard-clamps output to (0, 1), enforcing the physical
    constraint that occupancy is a ratio bounded by pile capacity.
    Values near 0 = almost empty zone; near 1 = almost fully occupied.

    Do not use this model's raw output for the consistency loss —
    the loss is computed in train.py after sigmoid using the predictions
    tensor, not the logits.
    """

    def __init__(self, cfg: Config, adj_sparse: torch.Tensor):
        super().__init__()

        mc  = cfg.model
        dc  = cfg.data

        # Internal sequence length after FeatureFusion
        seq_prime = dc.seq_len - mc.conv_kernel + 1

        # ---- Submodules ----------------------------------------------------
        self.fusion  = FeatureFusion(
            conv_kernel = mc.conv_kernel,
            n_features  = dc.n_features,
        )
        self.spatial = SpatialEncoder(
            adj_sparse     = adj_sparse,
            seq_len        = seq_prime,
            n_heads        = mc.gat_heads,
            dropout        = mc.dropout,
            gat_alpha      = mc.gat_alpha,
            residual_alpha = mc.residual_alpha,
        )
        self.temporal = TPADecoder(
            seq_len     = seq_prime,
            lstm_hidden = mc.lstm_hidden,
            lstm_layers = mc.lstm_layers,
            tpa_k       = mc.tpa_k,
        )

    def forward(
        self,
        occ: torch.Tensor,   # (B, N, S)
        prc: torch.Tensor,   # (B, N, S)
    ) -> torch.Tensor:
        """
        Returns:
            predictions : (B, N)  occupancy predictions in (0, 1)
        """
        fused          = self.fusion(occ, prc)           # (B, N, S')
        layer1, layer2 = self.spatial(fused)             # (B, N, S') each
        logits         = self.temporal(layer1, layer2)   # (B, N)
        predictions    = torch.sigmoid(logits)           # (B, N) ∈ (0, 1)
        return predictions


# ---------------------------------------------------------------------------
# Factory function — the only thing other files import
# ---------------------------------------------------------------------------

def build_model(cfg: Config, adj_sparse: torch.Tensor) -> STGAT:
    """
    Instantiates and returns an STGAT model on the correct device.

    This is the single entry point for model creation. Never instantiate
    STGAT directly in other files — always use build_model() so that
    device placement and config validation are guaranteed.

    Args:
        cfg        : Config object (fully validated)
        adj_sparse : sparse COO adjacency tensor from DataBundle.adj_sparse

    Returns:
        model : STGAT on cfg.system.device, ready for training
    """
    device = cfg.system.device

    # Move adj_sparse to device before passing to model
    # (GATLayer registers edges and mask as buffers — they need to be on
    # the right device at construction time, not just at forward time)
    adj_sparse = adj_sparse.to(device)

    model = STGAT(cfg, adj_sparse).to(device)

    # Parameter count summary
    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  [model] Built STGAT on {device}")
    print(f"  [model] Total parameters    : {total_params:,}")
    print(f"  [model] Trainable parameters: {trainable_params:,}")

    return model


# ---------------------------------------------------------------------------
# Self-test — run directly: python model.py
# Does not require the dataset — uses random tensors of the correct shape.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from config import Config, validate_config

    cfg = Config()
    validate_config(cfg)

    device = cfg.system.device
    N  = cfg.model.n_nodes       # 247
    S  = cfg.data.seq_len        # 12
    B  = 4                       # small batch for testing

    print(f"\nRunning model.py self-test on device: {device}")
    print(f"Input shape: ({B}, {N}, {S})\n")

    # Build a small random sparse adjacency for testing
    # (realistic density ~2% for 247 nodes)
    torch.manual_seed(42)
    adj_dense  = (torch.rand(N, N) < 0.02).float()
    adj_dense  = ((adj_dense + adj_dense.T) > 0).float()  # symmetrize
    adj_dense.fill_diagonal_(0)
    adj_sparse = adj_dense.to_sparse_coo()

    # Build model
    model = build_model(cfg, adj_sparse)

    # Forward pass with random inputs
    occ = torch.rand(B, N, S).to(device)
    prc = torch.rand(B, N, S).to(device)

    model.eval()
    with torch.no_grad():
        predictions = model(occ, prc)

    print(f"\n  Output shape : {tuple(predictions.shape)}  (expected: ({B}, {N}))")
    print(f"  Output range : [{predictions.min():.6f}, {predictions.max():.6f}]"
          f"  (expected: strictly within (0, 1))")
    print(f"  Output dtype : {predictions.dtype}")

    # Verify sigmoid constraint
    assert predictions.shape == (B, N), \
        f"Wrong output shape: {predictions.shape}"
    assert predictions.min() > 0.0 and predictions.max() < 1.0, \
        f"Output outside (0, 1): [{predictions.min()}, {predictions.max()}]"
    assert not torch.isnan(predictions).any(), \
        "NaN values in output — check model initialization."
    assert not torch.isinf(predictions).any(), \
        "Inf values in output — check model initialization."

    # Verify all parameters are trainable (catch ParameterDict regression)
    non_trainable = [
        name for name, p in model.named_parameters() if not p.requires_grad
    ]
    assert len(non_trainable) == 0, \
        f"Non-trainable parameters found: {non_trainable}"

    # Verify gradient flows through the entire model
    model.train()
    pred = model(occ, prc)
    loss = pred.mean()
    loss.backward()

    no_grad = [
        name for name, p in model.named_parameters()
        if p.requires_grad and p.grad is None
    ]
    if no_grad:
        print(f"\n  [Warning] Parameters with no gradient: {no_grad}")
    else:
        print(f"  Gradient flow check        : PASSED (all parameters received gradients)")

    print(f"  Output shape check         : PASSED")
    print(f"  Sigmoid constraint check   : PASSED")
    print(f"  Trainable parameters check : PASSED")
    print(f"\nmodel.py self-test complete.\n")