import torch
import torch.nn as nn
import math

class BiLSTMTransitionModel(nn.Module):
    """
    A bidirectional LSTM that processes sequences of embeddings (T' x D).
    We apply an exponential weighting near the boundary frames to emphasize transitions.
    Then we take the final hidden state (or a weighted average) as the output vector.
    """

    def __init__(self, input_dim=768, hidden_dim=128, num_layers=2, dropout=0.2, alpha=1.5):
        """
        Args:
            input_dim  : dimension of input embeddings (e.g., 768 for BEATs).
            hidden_dim : hidden dimension in LSTM.
            num_layers : LSTM stacked layers.
            dropout    : LSTM dropout (only used if num_layers > 1).
            alpha      : exponential weighting factor near boundaries.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # If num_layers=1, set dropout=0.0 to avoid warnings
        final_dropout = dropout if num_layers > 1 else 0.0

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=final_dropout,
            bidirectional=True,
            batch_first=True
        )

        # We'll do a final linear layer to produce an embedding of size hidden_dim
        # (since it's bidirectional => 2*hidden_dim)
        self.fc = nn.Linear(hidden_dim * 2, hidden_dim)

        # Exponential weighting factor
        self.alpha = alpha

    def forward(self, x):
        """
        x: [B, T, D] variable T across the batch.
        We'll do:
          1) LSTM -> [B, T, 2*hidden_dim]
          2) Weighted average along T dimension with an exponential weighting near boundaries
          3) fc -> final embedding [B, hidden_dim]
        """
        # Step 1: LSTM
        lstm_out, (h, c) = self.lstm(x)  # shape (B, T, 2*hidden_dim)

        B, T, _ = lstm_out.shape

        # Step 2: Exponential weighting near boundaries
        # w(t) = alpha^(- min(t, T-1-t))
        w = []
        for t in range(T):
            dist_left  = t
            dist_right = (T - 1) - t
            d = min(dist_left, dist_right)
            val = (self.alpha) ** (-d)
            w.append(val)
        # w is length T
        w = torch.tensor(w, dtype=torch.float32, device=lstm_out.device)  # shape (T,)
        w = w.unsqueeze(0).expand(B, T)  # shape (B, T)

        # Weighted sum
        w_3d = w.unsqueeze(-1)  # (B, T, 1)
        weighted = lstm_out * w_3d  # (B, T, 2*hidden_dim)

        sum_weighted = weighted.sum(dim=1)   # (B, 2*hidden_dim)
        sum_w = w.sum(dim=1, keepdim=True)   # (B, 1)
        avg_pooled = sum_weighted / sum_w    # (B, 2*hidden_dim)

        # Step 3: Final FC
        out = self.fc(avg_pooled)  # shape (B, hidden_dim)
        return out
