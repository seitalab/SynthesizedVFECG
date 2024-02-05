# https://github.com/sooftware/luna-transformer/blob/main/luna_transformer/encoder.py
# https://github.com/sooftware/luna-transformer/blob/main/luna_transformer/model.py
import math
from typing import Tuple
from argparse import Namespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

def quant_noise(module, p, block_size):
    """
    Wraps modules and applies quantization noise to the weights for
    subsequent quantization with Iterative Product Quantization as
    described in "Training with Quantization Noise for Extreme Model Compression"
    Args:
        - module: nn.Module
        - p: amount of Quantization Noise
        - block_size: size of the blocks for subsequent quantization with iPQ
    Remarks:
        - Module weights must have the right sizes wrt the block size
        - Only Linear, Embedding and Conv2d modules are supported for the moment
        - For more detail on how to quantize by blocks with convolutional weights,
          see "And the Bit Goes Down: Revisiting the Quantization of Neural Networks"
        - We implement the simplest form of noise here as stated in the paper
          which consists in randomly dropping blocks
    """

    # if no quantization noise, don't register hook
    if p <= 0:
        return module

    # supported modules
    assert isinstance(module, (nn.Linear, nn.Embedding, nn.Conv2d))

    # test whether module.weight has the right sizes wrt block_size
    is_conv = module.weight.ndim == 4

    # 2D matrix
    if not is_conv:
        assert module.weight.size(1) % block_size == 0, "Input features must be a multiple of block sizes"

    # 4D matrix
    else:
        # 1x1 convolutions
        if module.kernel_size == (1, 1):
            assert module.in_channels % block_size == 0, "Input channels must be a multiple of block sizes"
        # regular convolutions
        else:
            k = module.kernel_size[0] * module.kernel_size[1]
            assert k % block_size == 0, "Kernel size must be a multiple of block size"

    def _forward_pre_hook(mod, input):
        # no noise for evaluation
        if mod.training:
            if not is_conv:
                # gather weight and sizes
                weight = mod.weight
                in_features = weight.size(1)
                out_features = weight.size(0)

                # split weight matrix into blocks and randomly drop selected blocks
                mask = torch.zeros(in_features // block_size * out_features, device=weight.device)
                mask.bernoulli_(p)
                mask = mask.repeat_interleave(block_size, -1).view(-1, in_features)

            else:
                # gather weight and sizes
                weight = mod.weight
                in_channels = mod.in_channels
                out_channels = mod.out_channels

                # split weight matrix into blocks and randomly drop selected blocks
                if mod.kernel_size == (1, 1):
                    mask = torch.zeros(int(in_channels // block_size * out_channels), device=weight.device)
                    mask.bernoulli_(p)
                    mask = mask.repeat_interleave(block_size, -1).view(-1, in_channels)
                else:
                    mask = torch.zeros(weight.size(0), weight.size(1), device=weight.device)
                    mask.bernoulli_(p)
                    mask = mask.unsqueeze(2).unsqueeze(3).repeat(1, 1, mod.kernel_size[0], mod.kernel_size[1])

            # scale weights and apply mask
            mask = mask.to(torch.bool)  # x.bool() is not currently supported in TorchScript
            s = 1 / (1 - p)
            mod.weight.data = s * weight.masked_fill(mask, 0)

    module.register_forward_pre_hook(_forward_pre_hook)
    return module

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

class LinearUnifiedNestedAttention(nn.Module):

    def __init__(
        self, 
        emb_dim: int, 
        qkv_dim: int, 
        num_heads: int=8,
        num_pheads: int=8,
        dropout_rate: float=0.3,
        tie_kv: bool=False,        
        bias: bool=True, 
        batch_first: bool=True
    ) -> None:
        super(LinearUnifiedNestedAttention, self).__init__()
        """
        Args: 
            emb_dim (int): 
            qkv_dim (int): 
            num_heads (int): 
            num_pheads (int): 
            dropout_rate (float): 
            tie_kv (bool): 
            bias (bool): 
            batch_first (bool): 
        Returns:
            None
        """
        self.tie_kv = tie_kv
        self.batch_first = batch_first

        self.num_heads = num_heads
        self.num_pheads = num_pheads

        self.in_proj = nn.Linear(emb_dim, qkv_dim)
        self.in_proj_p = nn.Linear(emb_dim, qkv_dim)
        self.in_proj_c = nn.Linear(emb_dim, qkv_dim)

        self.head_dim = qkv_dim // num_heads
        self.phead_dim = qkv_dim // num_pheads
        assert (self.head_dim * num_heads == qkv_dim), "qkv_dim must be divisible by num_heads"
        assert (self.phead_dim * num_pheads == qkv_dim), "projected qkv_dim must be divisible by num_pheads"

        self.scaling = self.head_dim ** -0.5
        self.p_scaling = self.phead_dim ** -0.5

        self.q_proj = nn.Linear(qkv_dim, qkv_dim, bias=bias)
        self.pq_proj = nn.Linear(qkv_dim, qkv_dim, bias=bias)
        if tie_kv:
            self.c_proj = nn.Linear(qkv_dim, qkv_dim, bias=bias)
            self.pc_proj = nn.Linear(qkv_dim, qkv_dim, bias=bias)
        else:
            self.k_proj = nn.Linear(qkv_dim, qkv_dim, bias=bias)
            self.pk_proj = nn.Linear(qkv_dim, qkv_dim, bias=bias)
            self.v_proj = nn.Linear(qkv_dim, qkv_dim, bias=bias)
            self.pv_proj = nn.Linear(qkv_dim, qkv_dim, bias=bias)
        
        self.out_proj = nn.Linear(qkv_dim, emb_dim, bias=bias)
        self.out_proj_p = nn.Linear(qkv_dim, emb_dim)

        self.dropout_attn = nn.Dropout(dropout_rate)
        self.dropout_pcontext = nn.Dropout(dropout_rate)
        self._reset_params()
        
    def _reset_params(self) -> None:
        """
        Args:
            None
        Returns:
            None
        """
        gain = 1.0 / math.sqrt(2.0)
        nn.init.xavier_uniform_(self.q_proj.weight, gain=gain)
        nn.init.xavier_uniform_(self.pq_proj.weight, gain=gain)

        if self.tie_kv:
            nn.init.xavier_uniform_(self.pc_proj.weight, gain=gain)
            nn.init.xavier_uniform_(self.c_proj.weight, gain=gain)
        else:
            nn.init.xavier_uniform_(self.k_proj.weight, gain=gain)
            nn.init.xavier_uniform_(self.pk_proj.weight, gain=gain)
            nn.init.xavier_uniform_(self.v_proj.weight, gain=gain)
            nn.init.xavier_uniform_(self.pv_proj.weight, gain=gain)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.)

    def _compute_pcontext(self, pquery: Tensor, context: Tensor) -> Tensor:
        """
        Args:
            pquery (Tensor): Tensor of size(luna_context_len, batchsize, embed_dim)
            context (Tensor): Tensor of (luna_context_len, batchsize, embed_dim).
        Returns:
            pcontext (Tensor): Tensor of (luna_context_len, batchsize, embed_dim).
        """
        seqlen, bs, dim = context.size()
        p_len = pquery.size(0)

        if self.tie_kv:
            c = self.pc_proj(context)
            k = v = c.view(seqlen, bs * self.num_pheads, self.phead_dim) # -> N x B*H x K
        else:
            k = self.pk_proj(context)
            k = k.view(seqlen, bs * self.num_pheads, self.phead_dim) # -> N x B*H x K
            v = self.pv_proj(context)
            v = v.view(seqlen, bs * self.num_pheads, self.phead_dim) # -> N x B*H x K
        
        k = k.permute(1, 2, 0) # -> B*H x K x N
        v = v.transpose(0, 1) # -> B*H x N x K

        pq = self.pq_proj(pquery)
        pq = pq.view(p_len, bs * self.num_pheads, self.phead_dim) # -> L x B*H x K
        pq = pq.transpose(0, 1) * self.p_scaling # -> B*H x L x K

        pqk = torch.bmm(pq, k)
        pqk = self.dropout_pcontext(pqk)
        pqk = F.softmax(pqk, dim=-1) # -> B*H x L x N

        pc = torch.bmm(pqk, v) # -> B*H x L x K
        pc = pc.transpose(0, 1) # -> L x B*H x K
        pc = pc.contiguous().view(p_len, bs, dim)
        return pc

    def forward(
        self, 
        query: Tensor, 
        pquery: Tensor, 
        context: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            query (Tensor): 
                if batch_first, Tensor of (batchsize, seqlen, emb_dim) 
                else (seqlen, batchsize, emb_dim).
            pquery (Tensor): 
            context (Tensor): 
        Returns:
            attn (Tensor): 
            p_context (Tensor): 
            attn_weights (Tensor): 
        """
        if self.batch_first:
            query = query.transpose(0, 1)
            pquery = pquery.transpose(0, 1)
            context = context.transpose(0, 1)
        
        # Scale dim.
        query = self.in_proj(query)
        pquery = self.in_proj_p(pquery)
        context = self.in_proj_c(context)

        seqlen, bs, dim = query.size()

        p_context = self._compute_pcontext(pquery, context) # -> L x B*h x k

        q = self.q_proj(query) # -> N x B x D
        if self.tie_kv:
            c = self.c_proj(p_context)
            k = c.view(-1, bs * self.num_heads, self.head_dim).transpose(0, 1) # -> B*H x L x K
            v = c.view(-1, bs * self.num_heads, self.head_dim).transpose(0, 1) # -> B*H x L x K
        else:
            k = self.k_proj(p_context)
            k = k.view(-1, bs * self.num_heads, self.head_dim).transpose(0, 1) # -> B*H x L x K
            v = self.v_proj(p_context)
            v = v.view(-1, bs * self.num_heads, self.head_dim).transpose(0, 1) # -> B*H x L x K

        q = q * self.scaling
        q = q.contiguous().view(seqlen, bs * self.num_heads, self.head_dim)
        q = q.transpose(0, 1) # -> B*H x N x K

        attn_weights = torch.bmm(q, k.transpose(1, 2)) # -> B*H x N x L
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights_d = self.dropout_attn(attn_weights) # -> B*H x N x L

        attn = torch.bmm(attn_weights_d, v) # -> B*H x N x K
        attn = attn.transpose(0, 1)
        attn = attn.contiguous().view(seqlen, bs, dim)
        attn = self.out_proj(attn)

        p_context = self.out_proj_p(p_context)

        if self.batch_first:
            attn = attn.transpose(0, 1)
            p_context = p_context.transpose(0, 1)
            attn_weights = attn_weights.transpose(0, 1)

        return attn, p_context, attn_weights

class LunaTransformerBlock(nn.Module):

    def __init__(
        self,
        emb_dim: int,
        qkv_dim: int,
        num_heads: int,
        num_pheads: int, 
        mlp_dim: int,
        dropout_rate: float=0.3,
        attn_dropout_rate: float=0.2,
        pre_norm: bool=False,
        tie_kv: bool=False,
        batch_first: bool=True
    ) -> None:
        """
        Args:
            emb_dim (int): 
            qkv_dim (int): 
            num_heads (int): Number of heads.
            num_pheads (int): 
            mlp_dim (int): Size of dimension in feed forward layer.
            dropout_rate (float): Dropout rate for feed forward layer.
            attn_dropout_rate (float): Dropout rate for attention module.
            pre_norm (bool): 
            batch_first (bool): 
        Returns:
            None
        """
        super(LunaTransformerBlock, self).__init__()

        assert qkv_dim % num_heads == 0
        assert qkv_dim % num_pheads == 0

        self.pre_norm = pre_norm
        if pre_norm: 
            self.lnorm_pre = nn.LayerNorm(emb_dim)
            self.lnorm_pre_p = nn.LayerNorm(emb_dim)
            self.lnorm_ff_pre = nn.LayerNorm(emb_dim)
        else:
            self.lnorm_post = nn.LayerNorm(emb_dim)
            self.lnorm_post_p = nn.LayerNorm(emb_dim)
            self.lnorm_ff_post = nn.LayerNorm(emb_dim)

        self.dropout = nn.Dropout(dropout_rate)
        self.dropout_p = nn.Dropout(dropout_rate)

        self.luna_attention = LinearUnifiedNestedAttention(
            emb_dim, qkv_dim, num_heads, num_pheads, 
            dropout_rate=attn_dropout_rate, tie_kv=tie_kv,
            batch_first=batch_first
        )
        self.mlpblock = MLPBlock(emb_dim, mlp_dim, dropout_rate)

    def forward(self, input: Tensor, packed: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            input (Tensor): Input Tensor.
            packed (Tensor): Packed context.
        Returns:
            x (Tensor): 
            p_x (Tensor): 
        """
        residual = input
        p_residual = packed

        if self.pre_norm:
            input = self.lnorm_pre(input)
            packed = self.lnorm_pre_p(packed)
        
        x, p_x, _ = self.luna_attention(input, packed, input)

        x = self.dropout(x)
        p_x = self.dropout_p(p_x)

        x = residual + x
        p_x = p_residual + p_x

        if not self.pre_norm:
            x = self.lnorm_post(x)
            p_x = self.lnorm_post_p(p_x)
        
        residual = x
        if self.pre_norm:
            x = self.lnorm_ff_pre(x)
        
        # FFN layer.
        x = self.mlpblock(x)
        x = residual + x

        if not self.pre_norm:
            x = self.lnorm_ff_post(x)

        return x, p_x

class LunaTransformerModel(nn.Module):
    """
    Transformer encoder architecture applied Linear Unified Nested Attention (Luna).
    Luna was proposed in the paper "Luna: Linear Unified Nested Attention" (https://arxiv.org/abs/2106.01540.pdf)
    """
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        num_pheads:int,
        emb_dim: int,
        qkv_dim: int, 
        ff_dim: int, 
        out_dim: int,
        context_length: int,
        feat_select: str,
        seqlen: int,
        batch_first: bool=True,
    ) -> None:
        """
        Args:
            num_layers (int): Number of layers.
            num_heads (int): Number of heads in transformer encoder.
            num_pheads (int): 
            emb_dim (int): Size of each time step input.
            qkv_dim (int): 
            ff_dim (int): Size of feed forward module in transformer module.
            out_dim (int): Size of output.
            context_length (int): Length of packed context used for LUNA.
            batch_first (bool): 
        """
        super(LunaTransformerModel, self).__init__()

        self.emb_dim = emb_dim
        self.context_length = context_length

        self.embed_scale = math.sqrt(emb_dim)
        self.projected_embeds = nn.Parameter(
            torch.Tensor(context_length, emb_dim))
        nn.init.normal_(self.projected_embeds, mean=0.0, std=emb_dim ** -0.5)

        encoders = [
            LunaTransformerBlock(
                emb_dim=emb_dim, 
                qkv_dim=qkv_dim,
                num_heads=num_heads,
                num_pheads=num_pheads, 
                mlp_dim=ff_dim,
                batch_first=batch_first
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
        projected_embedded = self.projected_embeds * self.embed_scale
        projected_embedded = projected_embedded.unsqueeze(0).expand(
            feat.size(0), self.context_length, self.emb_dim)
        
        for encoder_layer in self.encoder_layers:
            feat, projected_embedded = encoder_layer(feat, projected_embedded)
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
        feat = feat.squeeze(-1) # -> bs, d_model
        feat = self.fc(feat) # -> bs, out_dim
        return feat

class LunaTransformer(nn.Module):

    def __init__(self, params: Namespace):
        super(LunaTransformer, self).__init__()

        seqlen = int(
            (params.max_duration * params.freq / params.downsample) / params.lin_chunk_len
        ) + 1 # +1 for token added during LinearEmbed.

        num_pheads = params.heads
        batch_first = True # Fixed
        self.backbone = LunaTransformerModel(
            num_layers=params.depth, 
            num_heads=params.heads, 
            num_pheads=num_pheads, 
            emb_dim=params.emb_dim, 
            qkv_dim=params.qkv_dim, 
            ff_dim=params.ff_dim, 
            out_dim=params.backbone_out_dim,
            context_length=params.luna_context_len,
            feat_select=params.feat_select,
            seqlen=seqlen,
            batch_first=batch_first
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.backbone(x)
