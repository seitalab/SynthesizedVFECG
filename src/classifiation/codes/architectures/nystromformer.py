import math
from argparse import Namespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

def pad_tensor(target_tensor: Tensor, pad_length: int) -> Tensor:
    """
    Args:
        target_tensor (Tensor): Tensor of size (bs, seqlen, dim1).
        pad_length (int): 
    Returns:
        padded_tensor (Tensor): Tensor of size (bs, seqlen + pad_length, dim1).
    """
    target_tensor = target_tensor.permute(0, 2, 1) # -> [bs, dim1, seqlen]
    padded_tensor = F.pad(target_tensor, (0, pad_length))
    padded_tensor = padded_tensor.permute(0, 2, 1) # -> [bs, seqlen, dim1, dim2]
    return padded_tensor

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

class NystromAttention(nn.Module):

    def __init__(
        self, 
        emb_dim: int,
        qkv_dim: int,
        num_heads: int, 
        num_landmarks: int,
        use_conv: bool=False,
        n_iter: int=6,
        init_option: str="original"
    ) -> None:
        super(NystromAttention, self).__init__()
        """
        Args:
            emb_dim (int): 
            qkv_dim (int): 
            num_heads (int): Number of heads.
            num_landmarks (int): 
            dropout_rate (float): Dropout rate for feed forward layer.
            use_conv (bool): 
            n_iter (int): 
            init_option (str): 
        Returns:
            None
        """

        self.num_heads = num_heads
        self.head_dim = qkv_dim // num_heads
        assert (self.head_dim * num_heads == qkv_dim), "qkv_dim must be divisible by num_heads"

        self.num_landmarks = num_landmarks

        self.dense_q = nn.Linear(emb_dim, qkv_dim)
        self.dense_k = nn.Linear(emb_dim, qkv_dim)
        self.dense_v = nn.Linear(emb_dim, qkv_dim)

        self.dense_out = nn.Linear(qkv_dim, emb_dim)

        self.n_iter = n_iter
        self.init_option = init_option

        self.use_conv = use_conv
        if use_conv:
            raise NotImplementedError
            kernel_size = ()
            padding = ()
            self.conv = nn.Conv2d(
                in_channels=self.num_heads, out_channels=self.num_heads,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
                groups=self.num_heads
            )

    def forward(self, x: Tensor)-> Tensor:
        """
        Args:
            x (Tensor): Tensor of size (batchsize, seqlen, emb_dim).
        Returns:
            x (Tensor): Tensor of size (batchsize, seqlen, emb_dim).
        """
        bs, seqlen, _ = x.size()

        if seqlen % self.num_landmarks > 0:
            pad_len = self.num_landmarks - (seqlen % self.num_landmarks)
            full_len = seqlen + pad_len
            x = pad_tensor(x, pad_len)
        else:
            full_len = seqlen

        q = self._split_heads(self.dense_q(x))
        k = self._split_heads(self.dense_k(x))
        v = self._split_heads(self.dense_v(x))

        q = q / math.sqrt(math.sqrt(self.head_dim)) # -> [bs, num_heads, full_len, head_dim].
        k = k / math.sqrt(math.sqrt(self.head_dim))
        
        if self.num_landmarks == full_len:
            qk = torch.matmul(q, k.transpose(-1, -2))
            attn = F.softmax(qk, dim=-1)
            x = torch.matmul(attn, v)
        else:
            q_landmarks = q.reshape(
                bs, self.num_heads, self.num_landmarks, 
                full_len // self.num_landmarks, self.head_dim
            ).mean(dim=-2)
            k_landmarks = k.reshape(
                bs, self.num_heads, self.num_landmarks, 
                full_len // self.num_landmarks, self.head_dim
            ).mean(dim=-2)
            qk_1 = torch.matmul(q, k_landmarks.transpose(-1, -2))
            kernel_1 = F.softmax(qk_1, dim=-1)
            qk_2 = torch.matmul(q_landmarks, k_landmarks.transpose(-1, -2))
            kernel_2 = F.softmax(qk_2, dim=-1)
            qk_3 = torch.matmul(q_landmarks, k.transpose(-1, -2))
            kernel_3 = F.softmax(qk_3, dim=-1)
            x = torch.matmul(
                torch.matmul(kernel_1, self._iterative_inv(kernel_2)),
                torch.matmul(kernel_3, v)
            )
        if self.use_conv:
            x = x + self.conv(v)

        # To original sequence length.
        x = x[:, :, :seqlen]
        # Combine heads.
        x = self._combine_heads(x)
        # qkv_dim -> emb_dim.
        x = self.dense_out(x)
        return x

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

    def _iterative_inv(self, kernel: Tensor) -> Tensor:
        """
        Args:
            kernel (Tensor): 
        Returns:
            V (Tensor): 
        """
        I = torch.eye(kernel.size(-1), device=kernel.device)
        K = kernel

        if self.init_option == "original":
            V = 1 / torch.max(torch.sum(K, dim=-2)) * K.transpose(-1, -2)
        else:
            V = 1 / torch.max(torch.sum(K, dim=-2), dim=-1).values[:, :, None, None] * K.transpose(-1, -2)
        
        for _ in range(self.n_iter):
            KV = torch.matmul(K, V)
            V = torch.matmul(
                0.25 * V, 13 * I - torch.matmul(
                    KV, 15 * I - torch.matmul(KV, 7 * I - KV)
                )
            )
        return V

class NystromformerBlock(nn.Module):

    def __init__(
        self, 
        emb_dim: int, 
        qkv_dim: int,
        num_heads: int,
        num_landmarks: int, 
        mlp_dim: int, 
        dropout_rate: float=0.3,
        attn_dropout_rate: float=0.2,
        pre_norm: bool=False,
    ) -> None:
        super(NystromformerBlock, self).__init__()
        """
        Args:
            emb_dim (int): 
            qkv_dim (int): 
            num_heads (int): Number of heads.
            num_landmarks (int): 
            mlp_dim (int): 
            dropout_rate (float): Dropout rate for feed forward layer.
            attn_dropout_rate (float): 
            pre_norm (bool): 
        Returns:
            None
        """

        assert qkv_dim % num_heads == 0
        self.head_dim = qkv_dim // num_heads

        self.pre_norm = pre_norm
        if pre_norm:
            self.lnorm_pre = nn.LayerNorm(emb_dim)
            self.lnorm_ff_pre = nn.LayerNorm(emb_dim)
        else:
            self.lnorm_post = nn.LayerNorm(emb_dim)
            self.lnorm_ff_post = nn.LayerNorm(emb_dim)

        self.dropout = nn.Dropout(dropout_rate)

        self.nystrom_attention = NystromAttention(
            emb_dim, 
            qkv_dim, 
            num_heads, 
            num_landmarks
        )
        self.proj = nn.Linear(qkv_dim, emb_dim)
        self.mlpblock = MLPBlock(emb_dim, mlp_dim, dropout_rate)

    def forward(self, input: Tensor) -> Tensor: 
        """
        Args:
            input (Tensor): Tensor of size (batchsize, seqlen, emb_dim).
        Returns:
            x (Tensor): Tensor of size (batchsize, seqlen, emb_dim).
        """
        residual = input

        if self.pre_norm:
            input = self.lnorm_pre(input)        

        x = self.nystrom_attention(input)
        x = self.dropout(x)
        x = residual + x

        if not self.pre_norm:
            x = self.lnorm_post(x)

        residual = x
        if self.pre_norm:
            x = self.lnorm_ff_pre(x)

        x = self.mlpblock(x)
        x = residual + x

        if not self.pre_norm:
            x = self.lnorm_ff_post(x)

        return x

class NystromformerModel(nn.Module):

    def __init__(
        self, 
        num_layers: int,
        num_heads: int,
        num_landmarks: int,
        emb_dim: int,
        qkv_dim: int,
        ff_dim: int, 
        out_dim: int,
        feat_select: str,
        seqlen: int
    ) -> None:
        """
        Args:
            num_layers (int): Number of layers.
            num_heads (int): Number of heads in transformer encoder.
            num_landmarks (int): Number of landmarks used in nystrom attention.
            emb_dim (int): Size of each time step input.
            qkv_dim (int): 
            ff_dim (int): Size of feed forward module in transformer module.
            out_dim (int): Size of output.
        Returns:
            None
        """
        super(NystromformerModel, self).__init__()

        encoders = [
            NystromformerBlock(
                emb_dim=emb_dim, 
                qkv_dim=qkv_dim,
                num_heads=num_heads,
                num_landmarks=num_landmarks,                
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
        feat = feat.squeeze(-1) # -> bs, d_model
        feat = self.fc(feat) # -> bs, out_dim
        return feat

class Nystromformer(nn.Module):

    def __init__(self, params: Namespace):
        super(Nystromformer, self).__init__()

        seqlen = int(
            (params.max_duration * params.freq / params.downsample) / params.lin_chunk_len
        ) + 1 # +1 for token added during LinearEmbed.

        self.backbone = NystromformerModel(
            num_layers=params.depth, 
            num_heads=params.heads, 
            num_landmarks=params.landmarks,
            emb_dim=params.emb_dim, 
            qkv_dim=params.qkv_dim, 
            ff_dim=params.ff_dim, 
            out_dim=params.backbone_out_dim,
            feat_select=params.feat_select,
            seqlen=seqlen
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.backbone(x)