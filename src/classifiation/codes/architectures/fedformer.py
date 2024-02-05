import sys

sys.path.append("../../repo/FEDformer")

# from layers.Embed import DataEmbedding, DataEmbedding_wo_pos
from layers.AutoCorrelation import AutoCorrelationLayer
from layers.FourierCorrelation import FourierBlock
from layers.MultiWaveletCorrelation import MultiWaveletTransform
from layers.Autoformer_EncDec import Encoder, EncoderLayer, my_Layernorm

from argparse import Namespace
import torch
import torch.nn as nn
from torch import Tensor

# class FEDformerEmbed(nn.Module):

#     def __init__(self, configs):
#         super(FEDformerEmbed, self).__init__()

#         # Embedding
#         # The series-wise connection inherently contains the sequential information.
#         # Thus, we can discard the position embedding of transformers.
#         self.enc_embedding = DataEmbedding_wo_pos(
#             configs.enc_in, 
#             configs.d_model, 
#             configs.embed, 
#             configs.freq,
#             configs.dropout
#         )

#     def forward(self, x):

#         x_enc, x_mark_enc = x
        
#         enc_out = self.enc_embedding(x_enc, x_mark_enc)

#         return enc_out

class FEDformerModel(nn.Module):
    """
    FEDformer performs the attention mechanism on frequency domain and achieved O(N) complexity.

    Modified from `https://github.com/MAZiqing/FEDformer/blob/master/models/FEDformer.py`.
    -> Only use encoder.
    """
    def __init__(self, configs):
        super(FEDformerModel, self).__init__()
        self.version = configs.version
        self.mode_select = configs.mode_select
        self.modes = configs.modes
        self.seq_len = configs.seq_len
        self.device = configs.device
        self.feat_select = configs.feat_select

        if configs.version == 'Wavelets':
            encoder_self_att = MultiWaveletTransform(
                ich=configs.d_model, L=configs.L, base=configs.base)
        else:
            encoder_self_att = FourierBlock(
                in_channels=configs.d_model,
                out_channels=configs.d_model,
                seq_len=self.seq_len,
                modes=configs.modes,
                mode_select_method=configs.mode_select
            )
        # Encoder
        enc_modes = int(min(configs.modes, configs.seq_len//2))
        print('enc_modes: {},'.format(enc_modes))

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        encoder_self_att,
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

        self.fc = nn.Linear(configs.d_model, configs.out_dim)

        if self.feat_select == "fc":
            self.fc_s = nn.Linear(self.seq_len, 1)

    def forward(self, x, enc_self_mask=None):

        # enc
        feat, _ = self.encoder(x, attn_mask=enc_self_mask)
        # feat = (bs, num_steps, d_model)

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

class FEDformer(nn.Module):

    def __init__(self, params: Namespace):
        super(FEDformer, self).__init__()

        seqlen = int(
            (params.max_duration * params.freq / params.downsample) / params.lin_chunk_len
        ) + 1 # +1 for token added during LinearEmbed.

        class Configs(object):
            modes = params.fedformer_mode

            mode_select = 'random'
            # mode_select = params.fedformer_mode_select

            # version = 'Fourier'
            # version = 'Wavelets'
            version= params.fedformer_version
            
            moving_avg = [12, 24]
            L = 1
            base = 'legendre'

            dropout = 0.05
            factor = 1

            seq_len = seqlen
            d_model = params.emb_dim
            n_heads = params.heads
            d_ff = params.ff_dim
            e_layers = params.depth
            activation = 'gelu'

            feat_select=params.feat_select
            out_dim=params.backbone_out_dim
            device=params.device

        configs = Configs()
        self.backbone = FEDformerModel(configs)

    def forward(self, x: Tensor) -> Tensor:

        return self.backbone(x)

