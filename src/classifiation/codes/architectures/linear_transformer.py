"""
Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention

Implementation from https://github.com/idiap/fast-transformers/tree/master/fast_transformers.
"""
from argparse import Namespace
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class FeatureMap(nn.Module):
    """Define the FeatureMap interface."""
    def __init__(self, query_dims):
        super().__init__()
        self.query_dims = query_dims

    def new_feature_map(self, device):
        """Create a new instance of this feature map. In particular, if it is a
        random feature map sample new parameters."""
        raise NotImplementedError()

    def forward_queries(self, x):
        """Encode the queries `x` using this feature map."""
        return self(x)

    def forward_keys(self, x):
        """Encode the keys `x` using this feature map."""
        return self(x)

    def forward(self, x):
        """Encode x using this feature map. For symmetric feature maps it
        suffices to define this function, but for asymmetric feature maps one
        needs to define the `forward_queries` and `forward_keys` functions."""
        raise NotImplementedError()

    @classmethod
    def factory(cls, *args, **kwargs):
        """Return a function that when called with the query dimensions returns
        an instance of this feature map.
        It is inherited by the subclasses so it is available in all feature
        maps.
        """
        def inner(query_dims):
            return cls(query_dims, *args, **kwargs)
        return inner

class ActivationFunctionFeatureMap(FeatureMap):
    """Define a feature map that is simply an element-wise activation
    function."""
    def __init__(self, query_dims, activation_function):
        super().__init__(query_dims)
        self.activation_function = activation_function

    def new_feature_map(self, device):
        return

    def forward(self, x):
        return self.activation_function(x)

elu_feature_map = ActivationFunctionFeatureMap.factory(
    lambda x: torch.nn.functional.elu(x) + 1
)

class MLPBlock(nn.Module):

    def __init__(
        self,
        emb_dim: int, 
        mlp_dim: int,
        dropout_rate: float
    ) -> None:
        """
        Args:
            mlp_dim (int): 
            out_dim (int): 
            dropout_rate (float)
        Returns:
            None
        """
        super(MLPBlock, self).__init__()
        
        self.dense1 = nn.Linear(emb_dim, mlp_dim)
        self.dense2 = nn.Linear(mlp_dim, emb_dim)
        self.drop1 = nn.Dropout(dropout_rate)
        self.drop2 = nn.Dropout(dropout_rate)

        nn.init.xavier_uniform_(self.dense1.weight)
        nn.init.normal_(self.dense1.bias, std=1e-6)

        nn.init.xavier_uniform_(self.dense2.weight)
        nn.init.normal_(self.dense2.bias, std=1e-6)

    def forward(self, input: Tensor) -> Tensor:
        """
        Args:
            input (Tensor): Tensor of size 
        Returns:
            x (Tensor): 
        """
        x = self.dense1(input)
        x = F.gelu(x)
        x = self.drop1(x)
        x = self.dense2(x)
        x = self.drop2(x)
        return x

class LinearAttention(nn.Module):

    def __init__(
        self,
        emb_dim: int,
        qkv_dim: int,
        num_heads: int=8,
        eps: float=1e-6,
        feature_map: Optional[Callable]=None,
        # batch_first: bool=True
    ) -> None:
        """
        Args:
            emb_dim (int): 
            qkv_dim (int): 
            eps (float): 
        Returns:
            None
        """
        super(LinearAttention, self).__init__()

        self.num_heads = num_heads
        self.head_dim = qkv_dim // num_heads
        assert self.head_dim * num_heads == qkv_dim, "qkv_dim must be divisible by num_heads"
        
        self.eps = eps

        self.dense1_q = nn.Linear(emb_dim, qkv_dim)
        self.dense1_k = nn.Linear(emb_dim, qkv_dim)
        self.dense1_v = nn.Linear(emb_dim, qkv_dim)
        self.dense2 = nn.Linear(qkv_dim, emb_dim)

        self.feature_map = (
            feature_map(qkv_dim) if feature_map else
            elu_feature_map(qkv_dim)
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): Tensor of size (batchsize, sequence_length, emb_dim).
        Returns:

        """
        bs, seqlen, _ = x.size()
        q = self.dense1_q(x).view(bs, seqlen, self.num_heads, self.head_dim)
        k = self.dense1_k(x).view(bs, seqlen, self.num_heads, self.head_dim)
        v = self.dense1_v(x).view(bs, seqlen, self.num_heads, self.head_dim)

        self.feature_map.new_feature_map(q.device)
        q = self.feature_map.forward_queries(q) # -> [bs, seqlen, n_head, head_dim]
        k = self.feature_map.forward_keys(k)  # -> [bs, seqlen, n_head, head_dim]

        kv = torch.einsum("nshd,nshm->nhmd", k, v)

        z = 1 / (torch.einsum("nlhd,nhd->nlh", q, k.sum(dim=1)) + self.eps)
        
        v = torch.einsum("nlhd,nhmd,nlh->nlhm", q, kv, z)
        x = v.contiguous().view(bs, seqlen, -1)

        x = self.dense2(x)
        return x
        
class LinearTransformerBlock(nn.Module):

    def __init__(
        self, 
        emb_dim: int, 
        qkv_dim: int, 
        num_heads: int,
        mlp_dim: int,
        pre_norm: bool=False,
        dropout_rate: float=0.3,
        # batch_first: bool=True
    ) -> None:
        """
        Args:

        Returns:

        """
        super(LinearTransformerBlock, self).__init__()

        self.pre_norm = pre_norm
        if pre_norm: 
            self.lnorm_pre = nn.LayerNorm(emb_dim)
            self.lnorm_ff_pre = nn.LayerNorm(emb_dim)
        else:
            self.lnorm_post = nn.LayerNorm(emb_dim)
            self.lnorm_ff_post = nn.LayerNorm(emb_dim)

        self.dropout = nn.Dropout(dropout_rate)

        self.linear_attention = LinearAttention(
            emb_dim, qkv_dim, num_heads)
        self.mlpblock = MLPBlock(emb_dim, mlp_dim, dropout_rate)

    def forward(self, input: Tensor) -> None:
        """
        Args:
            x (Tensor): 
        Returns:
            None
        """
        residual = input

        if self.pre_norm:
            input = self.lnorm_pre(input)
        
        x = self.linear_attention(input)
        x = self.dropout(x)
        x = residual + x

        if not self.pre_norm:
            x = self.lnorm_post(x)
        
        residual = x
        if self.pre_norm:
            x = self.lnorm_ff_pre(x)
        
        # FFN layer.
        x = self.mlpblock(x)
        x = residual + x

        if not self.pre_norm:
            x = self.lnorm_ff_post(x)

        return x

class LinearTransfomerModel(nn.Module):
    
    def __init__(
        self,
        num_layers: int,
        emb_dim: int,
        qkv_dim: int, 
        num_heads: int, 
        ff_dim: int, 
        out_dim: int,
        feat_select: str,
        seqlen: int,
        batch_first: bool=True
    ) -> None:
        """
        Args:
            num_layers (int): Number of layers.
            emb_dim (int): Size of each time step input.
            qkv_dim (int): 
            ff_dim (int): Size of feed forward module in transformer module.
            out_dim (int): Size of output.
            batch_first (bool): 
        """
        super(LinearTransfomerModel, self).__init__()

        encoders = [
            LinearTransformerBlock(
                emb_dim=emb_dim, 
                qkv_dim=qkv_dim,
                num_heads=num_heads,
                mlp_dim=ff_dim,
            ) for _ in range(num_layers)
        ]
        self.encoder_layers = nn.ModuleList(encoders)

        self.lnorm = nn.LayerNorm(emb_dim)
        self.fc = nn.Linear(emb_dim, out_dim)

        self.feat_select = feat_select
        if self.feat_select == "fc":
            self.fc_s = nn.Linear(seqlen, 1)

    def forward(self, feat: Tensor) -> Tensor:
        """
        Args:
            feat (Tensor): Tensor of size (batch_size, num_steps, emb_dim).
        Returns:
            feat (Tensor): Tensor of size (batch_size, backbone_out_dim).
        """

        for encoder_layer in self.encoder_layers:
            feat = encoder_layer(feat)
        feat = self.lnorm(feat) # -> bs, num_steps, emb_dim

        feat = feat.permute(0, 2, 1) # -> bs, emb_dim, num_steps
        if self.feat_select == "last": 
            feat = feat[:, :, 0] # -> bs, d_model, 1 (LRA: common_layers.py # L188)
        elif self.feat_select == "mean":
            feat = torch.mean(feat, dim=-1)
        elif self.feat_select == "fc":
            feat = self.fc_s(feat)
        else:
            raise NotImplementedError(
                f"{self.feat_select} not Implemented")
        feat = feat.squeeze(-1) # -> bs, emb_dim
        feat = self.fc(feat) # -> bs, out_dim
        return feat

class LinearTransformer(nn.Module):

    def __init__(self, params: Namespace):
        super(LinearTransformer, self).__init__()

        seqlen = int(
            (params.max_duration * params.freq / params.downsample) / params.lin_chunk_len
        ) + 1 # +1 for token added during LinearEmbed.


        self.backbone = LinearTransfomerModel(
            num_layers=params.depth,
            emb_dim=params.emb_dim, 
            qkv_dim=params.qkv_dim, 
            num_heads=params.heads,
            ff_dim=params.ff_dim, 
            out_dim=params.backbone_out_dim,
            feat_select=params.feat_select,
            seqlen=seqlen
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.backbone(x)