import torch
import torch.nn as nn
import math

class BiLSTMTransitionModel(nn.Module):
    """
    A bidirectional LSTM that processes sequences of embeddings (T' x D).
    We apply an exponential weighting near the boundary frames to emphasize transitions.
    Then we take the final hidden state (or a weighted average) as the output vector.
    """

    def __init__(self, input_dim=768, hidden_dim=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim,
                            num_layers=num_layers,
                            dropout=dropout,
                            bidirectional=True,
                            batch_first=True)

        # We'll do a final linear layer to produce an embedding
        # dimension the same as input_dim or hidden_dim as you prefer
        # If we want a final embedding of size hidden_dim, that's typical for triplet.
        # Let's do hidden_dim * 2 for bidirectional => output_dim = hidden_dim
        self.fc = nn.Linear(hidden_dim * 2, hidden_dim)

        # Exponential weighting factor
        self.alpha = 1.5  # tune as needed

    def forward(self, x):
        """
        x: [B, T, D] variable T across the batch, so we might need to pad or
           handle in a collate_fn if needed. If your DataLoader uses a custom collate,
           you can handle variable seq lengths. Alternatively, pad to max T in the batch.

        We'll do:
        1) LSTM -> [B, T, 2*hidden_dim]
        2) Weighted average along T dimension with an exponential weighting near the boundary
        3) fc -> final embedding [B, hidden_dim]
        """
        # x shape: (B, T, D)
        # LSTM
        lstm_out, (h, c) = self.lstm(x)  # (B, T, 2*hidden_dim)

        # Exponential weighting near boundary:
        # Suppose T can vary. We'll create a weight w(t) that is larger near t=0 or t=T-1.
        # Because we used "two-stage embeddings" => anchor is near the end, positive is near the start,
        # so we actually want to emphasize the entire sequence but prefer boundaries.
        # We'll do a symmetrical weighting that peaks near both ends, e.g. w(t) = alpha^(-distance_from_edge).
        B, T, _ = lstm_out.shape

        # Build a weight matrix shape (B, T) or (1, T)
        # We'll do: dist_from_left = t, dist_from_right = (T-1 - t)
        # pick min dist => we want the boundary is min(t, T-1 - t).
        # w(t) = alpha ** (- min(t, T-1 - t))
        # to keep it simpler, we do it for each sample if T is the same across batch or each in a loop if variable.
        # If your batch has variable T, use "padding and a mask." We'll do a simple approach for uniform T batch.

        # For a variable T approach, you'd probably handle weighting in a custom collate or code that does the loop.
        # We'll assume T is uniform for the batch or you can just do the max and partial weighting for all.
        w = []
        for t in range(T):
            dist_left = t
            dist_right = (T - 1) - t
            d = min(dist_left, dist_right)
            val = (self.alpha) ** (-d)
            w.append(val)
        # w is length T
        w = torch.tensor(w, dtype=torch.float32, device=lstm_out.device)  # (T,)
        w = w.unsqueeze(0).expand(B, T)  # (B, T) broadcast across batch

        # Weighted sum
        # We have lstm_out: (B, T, 2*hidden_dim), we want to multiply each time step by w(t)
        # => do broadcast w: (B, T, 1)
        w_3d = w.unsqueeze(-1)  # (B, T, 1)
        weighted = lstm_out * w_3d  # (B, T, 2*hidden_dim)

        # Weighted average across T
        sum_weighted = weighted.sum(dim=1)   # (B, 2*hidden_dim)
        sum_w = w.sum(dim=1, keepdim=True)   # (B, 1)
        avg_pooled = sum_weighted / sum_w    # (B, 2*hidden_dim)

        # Final FC
        out = self.fc(avg_pooled)  # (B, hidden_dim)
        return out
