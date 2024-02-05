from typing import Any

import torch
import torch.nn as nn

class BiGRU(nn.Module):

    def __init__(
        self,
        depth: int,
        hidden: int,
        num_lead: int = 12,
        backbone_out_dim: int = 512,
    ) -> None:
        super(BiGRU, self).__init__()

        self.gru_hidden = hidden
        self.bigru = nn.GRU(
            num_lead, 
            hidden, 
            num_layers=depth, 
            bidirectional=True
        )

        self.fc = nn.Linear(hidden*2, backbone_out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: batch_size, num_lead, seqlen -> seqlen, batch_size, num_lead
        x = x.permute(2, 0, 1)
        feat, _ = self.bigru(x)

        # feat: seqlen, batch_size, lstm_hidden*2 -> batch_size, lstm_hidden*2, seqlen
        feat = feat.permute(1, 2, 0)
        feat = nn.AdaptiveAvgPool1d(1)(feat)
        feat = feat.squeeze(-1)
        feat = self.fc(feat)
        return feat

class VarDepthGRU(nn.Module):

    def __init__(self, params, num_lead: int):
        super(VarDepthGRU, self).__init__()

        self.rnn_model = BiGRU(
            params.rnn_depth, 
            params.rnn_hidden,
            num_lead,
            params.backbone_out_dim
        )

    def forward(self, x):
        if x.dim() == 2:
            # Assuming direct signal input.
            # Array of shape (bs, seqlen).
            x = x.unsqueeze(1)
        elif x.dim() == 3:
            # Assuming embedding as input.
            # Array of shape (bs, seqlen, emb_dim).
            x = x.permute(0, 2, 1)
        else:
            raise
        return self.rnn_model(x)