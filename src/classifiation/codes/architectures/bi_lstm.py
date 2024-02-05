from typing import Any

import torch
import torch.nn as nn

class BiLSTM(nn.Module):

    def __init__(
        self,
        lstm_depth: int,
        lstm_hidden: int,
        num_lead: int = 12,
        backbone_out_dim: int = 512,
    ) -> None:
        super(BiLSTM, self).__init__()

        self.lstm_hidden = lstm_hidden
        self.bilstm = nn.LSTM(
            num_lead, 
            lstm_hidden, 
            num_layers=lstm_depth, 
            bidirectional=True
        )

        self.fc = nn.Linear(lstm_hidden*2, backbone_out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: batch_size, num_lead, seqlen -> seqlen, batch_size, num_lead
        x = x.permute(2, 0, 1)
        feat, _ = self.bilstm(x)

        # feat: seqlen, batch_size, lstm_hidden*2 -> batch_size, lstm_hidden*2, seqlen
        feat = feat.permute(1, 2, 0)
        feat = nn.AdaptiveAvgPool1d(1)(feat)
        feat = feat.squeeze(-1)
        feat = self.fc(feat)
        return feat

class VarDepthLSTM(nn.Module):

    def __init__(self, params, num_lead: int):
        super(VarDepthLSTM, self).__init__()

        self.rnn_model = BiLSTM(
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