from argparse import Namespace

import torch
from torch import Tensor
import torch.nn as nn

class LinearEmbed(nn.Module):

    def __init__(
        self, 
        params: Namespace, 
        add_cls_token: bool=True
    ) -> None:
        super(LinearEmbed, self).__init__()

        self.num_lead = params.num_lead
        self.chunk_len = int(params.lin_chunk_len)

        chunk_dim = int(self.num_lead * self.chunk_len)
        self.embed = nn.Linear(chunk_dim, params.emb_dim)

        self.add_cls_token = add_cls_token

    def forward(self, x: Tensor):
        """
        Args:
            x (torch.Tensor): Tensor of size (batch_size, num_lead, seqlen).
        Returns:
            feat (torch.Tensor): Tensor of size (batch_size, num_chunks, emb_dim).
        """
        x = x.unsqueeze(1)
        assert x.size(1) == self.num_lead
        assert x.size(2) % self.chunk_len == 0

        bs = x.size(0)
        num_chunks = x.size(2) // self.chunk_len
        # batch_size, num_lead, num_chunks, chunk_len
        x = torch.reshape(x, (bs, self.num_lead, num_chunks, self.chunk_len))
        x = x.permute(0, 2, 1, 3)

        # batch_size, num_chunks, num_lead * chunk_len
        x = torch.reshape(x, (bs, num_chunks, -1))


        # ADD CLS Token.
        if self.add_cls_token:
            cls_token = torch.zeros(bs, 1, x.size(2)).to(x.device)
            x = torch.cat((cls_token, x), dim=1)

        feat = self.embed(x)
        return feat
    
class TransformerModel(nn.Module):

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        d_model: int,
        ff_dim: int,
        out_dim: int,
        feat_select: str,
        seqlen: int
    ) -> None:
        """
        Args:
            num_layers (int):
            num_heads (int): Number of heads in transformer encoder.
            d_model (int): Size of each time step input.
            ff_dim (int): Size of feed forward module in transformer module.
            out_dim (int): 
            feat_select (str): Multistep -> single feature method.
            seqlen (int): Length of input feature.
        """
        super(TransformerModel, self).__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=num_heads, 
            activation="gelu",
            dim_feedforward=ff_dim, batch_first=True)

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers)

        self.fc = nn.Linear(d_model, out_dim)

        self.feat_select = feat_select
        if self.feat_select == "fc":
            self.fc_s = nn.Linear(seqlen, 1)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): Tensor of size (batch_size, num_steps, d_model).
        Returns:
            feat (Tensor): Tensor of size (batch_size, backbone_out_dim).
        """

        feat = self.transformer_encoder(x) # -> bs, num_steps, d_model

        feat = feat.permute(0, 2, 1) # -> bs, d_model, num_steps
        if self.feat_select == "last": 
            feat = feat[:, :, 0] # -> bs, d_model, 1 (LRA: common_layers.py # L188)
        elif self.feat_select == "mean":
            feat = torch.mean(feat, dim=-1)
        elif self.feat_select == "fc":
            feat = self.fc_s(feat)
        else:
            raise NotImplementedError(
                f"{self.feat_select} not Implemented")
        feat = feat.squeeze(-1) # -> bs, d_model

        feat = self.fc(feat) # -> bs, out_dim
        return feat

class Transformer(nn.Module):

    def __init__(self, params: Namespace):
        super(Transformer, self).__init__()

        seqlen = int(
            (params.max_duration * params.freq / params.downsample) / params.lin_chunk_len
        ) + 1 # +1 for token added during LinearEmbed.

        self.backbone = TransformerModel(
            num_layers=params.depth, 
            num_heads=params.heads, 
            d_model=params.emb_dim, 
            ff_dim=params.ff_dim, 
            out_dim=params.backbone_out_dim,
            feat_select=params.feat_select,
            seqlen=seqlen
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.backbone(x)