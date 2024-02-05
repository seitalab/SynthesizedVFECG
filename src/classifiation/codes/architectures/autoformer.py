import sys

sys.path.append("../../repo/FEDformer")

from layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from layers.Autoformer_EncDec import Encoder, EncoderLayer, my_Layernorm

from argparse import Namespace
import torch
import torch.nn as nn
from torch import Tensor


class AutoformerModel(nn.Module):

    """
    Autoformer is the first method to achieve the series-wise connection,
    with inherent O(LlogL) complexity
    """
    def __init__(self, configs):
        super(AutoformerModel, self).__init__()
        self.seq_len = configs.seq_len
        self.device = configs.device
        self.feat_select = configs.feat_select
        self.output_attention = configs.output_attention

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(
                            False, 
                            configs.factor, 
                            attention_dropout=configs.dropout,
                            output_attention=configs.output_attention
                        ),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model)
        )

        # Out
        self.fc = nn.Linear(configs.d_model, configs.out_dim)

        if self.feat_select == "fc":
            self.fc_s = nn.Linear(self.seq_len, 1)

    def forward(self, x, enc_self_mask=None):

        # enc
        feat, _ = self.encoder(x, attn_mask=enc_self_mask)

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

        feat = self.fc(feat)
        return feat
    
class Autoformer(nn.Module):

    def __init__(self, params: Namespace):
        super(Autoformer, self).__init__()

        seqlen = int(
            (params.max_duration * params.freq / params.downsample) / params.lin_chunk_len
        ) + 1 # +1 for token added during LinearEmbed.

        class Configs(object):

            seq_len = seqlen
            d_model = params.emb_dim
            dropout = 0.05
            factor = 1
            n_heads = params.heads
            d_ff = params.ff_dim
            e_layers = params.depth
            activation = 'gelu'
            
            moving_avg = [12, 24]
            output_attention = True

            feat_select=params.feat_select
            out_dim=params.backbone_out_dim
            device=params.device

        configs = Configs()
        self.backbone = AutoformerModel(configs)

    def forward(self, x: Tensor) -> Tensor:

        return self.backbone(x)

