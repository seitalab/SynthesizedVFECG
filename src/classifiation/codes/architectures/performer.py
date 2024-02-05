"""
"""
import math
import functools
from argparse import Namespace
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import repeat

ATTN_TYPES = ["no_projection", "generalized_attn", "softmax_attn"]

def softmax_kernel(
    data: Tensor,
    *,
    projection_matrix: Tensor,
    is_query: bool,
    normalize_data: bool=True,
    eps: float=1e-4
):
    """
    Args:
        data (Tensor): Tensor of size (bs, seqlen, num_heads, head_dim).
        *,
        projection_matrix (Tensor): Tensor of size (qkv_dim, head_dim).
        is_query (bool): 
        normalize_data (bool): 
        eps (float):
    Returns:

    """
    bs, seqlen, *_ = data.shape
    data_normalizer = (data.size(-1) ** -0.25) if normalize_data else 1.

    ratio = (projection_matrix.size(0) ** -0.5)

    projection = repeat(
        projection_matrix, 'j d -> b h j d', b=bs, h=seqlen)
    projection = projection.type_as(data) # -> [bs, seqlen, qkv_dim, head_dim]

    data_dash = torch.einsum(
        '...id,...jd->...ij', 
        (data_normalizer * data), # -> [bs, seqlen, num_heads, head_dim]
        projection # -> [bs, seqlen, qkv_dim, head_dim]
    )

    diag_data = data ** 2
    diag_data = torch.sum(diag_data, dim=-1)
    diag_data = (diag_data / 2.0) * (data_normalizer ** 2)
    diag_data = diag_data.unsqueeze(dim=-1)

    if is_query:
        data_dash = ratio * (
            torch.exp(data_dash - diag_data -
                    torch.amax(data_dash, dim=-1, keepdim=True).detach()) + eps)
    else:
        data_dash = ratio * (
            torch.exp(data_dash - diag_data -\
                 torch.amax(data_dash, dim=(-1, -2), 
                 keepdim=True).detach()
            ) + eps)
    return data_dash.type_as(data)

def sincos_softmax_kernel(
    data: Tensor,
    *,
    projection_matrix: Tensor,
    normalize_data: bool=True,
): 
    """
    Args:

    Returns:

    """
    bs, seqlen, *_ = data.shape
    data_normalizer = (data.size(-1) ** -0.25) if normalize_data else 1.

    ratio = (projection_matrix.size(0) ** -0.5)
    
    projection = repeat(
        projection_matrix, 'j d -> b h j d', b = bs, h = seqlen)
    projection = projection.type_as(data)

    data_dash = torch.einsum(
        '...id,...jd->...ij', 
        (data_normalizer * data), 
        projection
    )
    data_dash_cos = ratio * torch.cos(data_dash)
    data_dash_sin = ratio * torch.sin(data_dash)
    data_dash = torch.cat((data_dash_cos, data_dash_sin), dim=-1)

    diag_data = data ** 2
    diag_data = torch.sum(diag_data, dim=-1)
    diag_data = (diag_data / 2.0) * (data_normalizer ** 2)
    diag_data = diag_data.unsqueeze(dim=-1)

    data_renormalizer = torch.amax(diag_data, dim=(-1, -2), keepdims=True)
    diag_data = diag_data - data_renormalizer
    diag_data = torch.exp(diag_data)
    data_prime = data_dash * diag_data
    return data_prime

def generalized_kernel(
    data: Tensor,
    *, 
    projection_matrix: Tensor,
    kernel_fn=nn.ReLU(),
    kernel_epsilon: float=0.001,
    normalize_data: bool=True,
) -> Tensor:
    """
    Args:

    Returns:

    """
    bs, seqlen, *_ = data.size()

    data_normalizer = (data.size(-1) ** -0.25) if normalize_data else 1.

    if projection_matrix is None:
        return kernel_fn(data_normalizer * data) + kernel_epsilon

    projection = repeat(
        projection_matrix, 'j d -> b h j d', b=bs, h=seqlen)
    projection = projection.type_as(data)

    data_dash = torch.einsum(
        '...id,...jd->...ij', 
        (data_normalizer * data), 
        projection
    )
    data_prime = kernel_fn(data_dash) + kernel_epsilon
    return data_prime

def orthogonal_matrix_chunk(cols: int, device: Optional[str]=None) -> Tensor:
    """
    Args:

    Returns:

    """
    unstructured_block = torch.randn((cols, cols), device=device)
    q, r = torch.linalg.qr(unstructured_block.cpu(), mode='reduced')
    q, r = map(lambda t: t.to(device), (q, r))
    return q.t()

def gaussian_orthogonal_random_matrix(
    nb_rows: int, 
    nb_cols: int, 
    scaling: float = 0, 
    device=None
) -> Tensor:
    """
    Args:
        nb_rows (int): 
        nb_cols (int): 
        scaling (float): 
        device
    Returns:

    """
    nb_full_blocks = int(nb_rows / nb_cols)

    block_list = []

    for _ in range(nb_full_blocks):
        q = orthogonal_matrix_chunk(nb_cols, device=device)
        block_list.append(q)

    remaining_rows = nb_rows - nb_full_blocks * nb_cols
    if remaining_rows > 0:
        q = orthogonal_matrix_chunk(nb_cols, device=device)
        block_list.append(q[:remaining_rows])

    final_matrix = torch.cat(block_list)

    if scaling == 0:
        multiplier = torch.randn(
            (nb_rows, nb_cols), device=device).norm(dim = 1)
    elif scaling == 1:
        multiplier = math.sqrt(
            (float(nb_cols))) * torch.ones((nb_rows,), 
            device=device)
    else:
        raise ValueError(f'Invalid scaling {scaling}')

    return torch.diag(multiplier) @ final_matrix

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

class PerformerAttention(nn.Module):

    def __init__(
        self, 
        emb_dim: int, 
        qkv_dim: int, 
        num_heads: int,
        ortho_scaling = 0, 
        kernel_fn = nn.ReLU(), 
        attn_type: str="softmax_attn"
    ) -> None:
        """
        Args:

        Returns:

        """
        super(PerformerAttention, self).__init__()

        self.num_heads = num_heads
        self.head_dim = qkv_dim // num_heads
        assert self.head_dim * num_heads == qkv_dim, "qkv_dim must be divisible by num_heads"

        self.attn_type = attn_type
        assert attn_type in ATTN_TYPES

        self.create_projection = functools.partial(
            gaussian_orthogonal_random_matrix, 
            nb_rows = qkv_dim, 
            nb_cols = self.head_dim, 
            scaling = ortho_scaling,
        )
        projection_matrix = self.create_projection()
        self.register_buffer('projection_matrix', projection_matrix)
        self.kernel_fn = kernel_fn

        self.dense_q = nn.Linear(emb_dim, qkv_dim)
        self.dense_k = nn.Linear(emb_dim, qkv_dim)
        self.dense_v = nn.Linear(emb_dim, qkv_dim)

        self.dense_out = nn.Linear(qkv_dim, emb_dim)    

    def forward(self, x: Tensor) -> Tensor:
        """
        Args: 

        Returns:

        """
        q = self._split_heads(self.dense_q(x))
        k = self._split_heads(self.dense_k(x))
        v = self._split_heads(self.dense_v(x))

        if self.attn_type == "no_projection":
            q = q.softmax(dim = -1)
            k = k.softmax(dim = -2)
        elif self.attn_type == "generalized_attn":
            create_kernel = functools.partial(
                generalized_kernel, 
                kernel_fn=self.kernel_fn, 
                projection_matrix=self.projection_matrix, 
            )
            q, k = map(create_kernel, (q, k))
        elif self.attn_type == "softmax_attn":
            create_kernel = functools.partial(
                softmax_kernel, 
                projection_matrix=self.projection_matrix
            )
            q = create_kernel(q, is_query = True)
            k = create_kernel(k, is_query = False)
        else:
            raise NotImplementedError
        
        out = self.linear_attn(q, k, v)

        out = self._combine_heads(out)
        out = self.dense_out(out)
        return out        

    def linear_attn(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        """
        Args:
            q (Tensor): 
            k (Tensor): 
            v (Tensor): 
        Returns:
            out (Tensor): 
        """
        k_cumsum = k.sum(dim = -2)
        D_inv = 1. / torch.einsum('...nd,...d->...n', q, k_cumsum.type_as(q))
        context = torch.einsum('...nd,...ne->...de', k, v)
        out = torch.einsum('...de,...nd,...n->...ne', context, q, D_inv)
        return out

    def _split_heads(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): Tensor of size (bs, seqlen, qkv_dim).
        Returns:
            x (Tensor): Tensor of size (bs, num_heads, seqlen, head_dim).
        """
        bs, seqlen, _ = x.size()
        x = x.reshape(bs, seqlen, self.num_heads, self.head_dim)
        x = x.transpose(1, 2)
        return x

    def _combine_heads(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): Tensor of size (bs, num_heads, seqlen, head_dim).
        Returns:
            x (Tensor): Tensor of size (bs, seqlen, qkv_dim).
        """
        x = x.transpose(1, 2)
        bs, seqlen, _, _ = x.size()
        x = x.reshape(bs, seqlen, self.num_heads*self.head_dim)
        return x

class PerformerBlock(nn.Module):

    def __init__(
        self, 
        emb_dim: int, 
        qkv_dim: int, 
        num_heads: int,
        mlp_dim: int,
        pre_norm: bool=False,
        dropout_rate: float=0.3,
        attn_type: str="softmax_attn",
        # batch_first: bool=True
    ) -> None:
        """
        Args:

        Returns:

        """
        super(PerformerBlock, self).__init__()

        self.pre_norm = pre_norm
        if pre_norm: 
            self.lnorm_pre = nn.LayerNorm(emb_dim)
            self.lnorm_ff_pre = nn.LayerNorm(emb_dim)
        else:
            self.lnorm_post = nn.LayerNorm(emb_dim)
            self.lnorm_ff_post = nn.LayerNorm(emb_dim)

        self.dropout = nn.Dropout(dropout_rate)

        self.linear_attention = PerformerAttention(
            emb_dim, qkv_dim, num_heads, attn_type=attn_type)
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

class PerformerModel(nn.Module):

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
        batch_first: bool=True,
        attn_type: str="softmax_attn",
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
        super(PerformerModel, self).__init__()

        encoders = [
            PerformerBlock(
                emb_dim=emb_dim, 
                qkv_dim=qkv_dim,
                num_heads=num_heads,
                mlp_dim=ff_dim,
                attn_type=attn_type
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
            feat (Tensor): Tensor of size (batch_size, num_steps, d_model).
        Returns:
            feat (Tensor): Tensor of size (batch_size, backbone_out_dim).
        """

        for encoder_layer in self.encoder_layers:
            feat = encoder_layer(feat)
        feat = self.lnorm(feat) # -> bs, num_steps, d_model

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

class Performer(nn.Module):

    def __init__(self, params: Namespace):
        super(Performer, self).__init__()

        seqlen = int(
            (params.max_duration * params.freq / params.downsample) / params.lin_chunk_len
        ) + 1 # +1 for token added during LinearEmbed.

        self.backbone = PerformerModel(
            num_layers=params.depth,
            emb_dim=params.emb_dim, 
            qkv_dim=params.qkv_dim, 
            num_heads=params.heads,
            ff_dim=params.ff_dim, 
            out_dim=params.backbone_out_dim,
            attn_type=params.attn_type,
            feat_select=params.feat_select,
            seqlen=seqlen
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.backbone(x)