"""
Based on `https://github.com/tianzhou2011/FiLM/blob/main/models/FiLM.py`
"""

import sys

sys.path.append("../../repo/FiLM")

from argparse import Namespace
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
from scipy import signal
from scipy import special as ss
from utils.op import transition
import opt_einsum as oe

contract = oe.contract


class HiPPO_LegT(nn.Module):

    def __init__(self, N, device, dt=1.0, discretization='bilinear'):
        """
        N: the order of the HiPPO projection
        dt: discretization step size - should be roughly inverse to the length of the sequence
        """
        super(HiPPO_LegT,self).__init__()

        self.device = device

        self.N = N
        A, B = transition('lmu', N)
        C = np.ones((1, N))
        D = np.zeros((1,))
        # dt, discretization options
        A, B, _, _, _ = signal.cont2discrete((A, B, C, D), dt=dt, method=discretization)

        B = B.squeeze(-1)

        self.register_buffer('A', torch.Tensor(A).to(device)) 
        self.register_buffer('B', torch.Tensor(B).to(device)) 
        vals = np.arange(0.0, 1.0, dt)
        self.register_buffer('eval_matrix',  torch.Tensor(
            ss.eval_legendre(np.arange(N)[:, None], 1 - 2 * vals).T).to(device))
    
    def forward(self, inputs):  # torch.Size([128, 1, 1]) -
        """
        inputs : (length, ...)
        output : (length, ..., N) where N is the order of the HiPPO projection
        """

        c = torch.zeros(inputs.shape[:-1] + tuple([self.N])).to(self.device)  # torch.Size([1, 256])
        cs = []
        for f in inputs.permute([-1, 0, 1]):
            f = f.unsqueeze(-1)
            # f: [1,1]
            new = f @ self.B.unsqueeze(0) # [B, D, H, 256]
            c = F.linear(c, self.A) + new
            # c = [1,256] * [256,256] + [1, 256]
            cs.append(c)
        return torch.stack(cs, dim=0)

    def reconstruct(self, c):
        a = (self.eval_matrix @ c.unsqueeze(-1)).squeeze(-1)
        return (self.eval_matrix @ c.unsqueeze(-1)).squeeze(-1)


################################################################

    

class SpectralConv1d(nn.Module):
    def __init__(
        self, 
        in_channels, 
        out_channels,
        seq_len, 
        modes1,
        compression=0,
        ratio=0.5,
        mode_type=0
    ):
        super(SpectralConv1d, self).__init__()
            

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.compression = compression
        self.ratio = ratio 
        self.mode_type=mode_type
        if self.mode_type ==1:
            #modes2=modes1-10000
            modes2 = modes1
            self.modes2 =min(modes2, seq_len//2)
            self.index0 = list(range(0, int(ratio*min(seq_len//2, modes2))))
            self.index1 = list(range(len(self.index0),self.modes2))
            np.random.shuffle(self.index1)
            self.index1 = self.index1[:min(seq_len//2,self.modes2)-int(ratio*min(seq_len//2, modes2))]
            self.index = self.index0+self.index1
            self.index.sort()
        elif self.mode_type > 1:
            #modes2=modes1-1000
            modes2 = modes1
            self.modes2 =min(modes2, seq_len//2)
            self.index = list(range(0, seq_len//2))
            np.random.shuffle(self.index)
            self.index = self.index[:self.modes2]
        else:
            self.modes2 =min(modes1, seq_len//2)
            self.index = list(range(0, self.modes2))

        self.scale = (1 / (in_channels*out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, len(self.index), dtype=torch.cfloat))
        if self.compression > 0:
            print('compressed version')
            self.weights0 = nn.Parameter(self.scale * torch.rand(in_channels,self.compression,dtype=torch.cfloat))
            self.weights1 = nn.Parameter(self.scale * torch.rand(self.compression,self.compression, len(self.index), dtype=torch.cfloat))
            self.weights2 = nn.Parameter(self.scale * torch.rand(self.compression,out_channels, dtype=torch.cfloat))
        #print(self.modes2)
        

    def forward(self, x):
        B, H, E, N = x.shape
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x)
        #pdb.set_trace()
        # Multiply relevant Fourier modes
        out_ft = torch.zeros(B,H, self.out_channels, x.size(-1)//2 + 1,  device=x.device, dtype=torch.cfloat)
        #a = x_ft[:, :,:, :self.modes1]
        #out_ft[:, :,:, :self.modes1] = torch.einsum("bjix,iox->bjox", a, self.weights1)
        if self.compression ==0:
            if self.modes1>1000:
                for wi, i in enumerate(self.index):
                    #print(self.index)
                    #print(out_ft.shape)
                    out_ft[:, :, :, i] = torch.einsum('bji,io->bjo',(x_ft[:, :, :, i], self.weights1[:, :,wi]))
            else:
                a = x_ft[:, :,:, :self.modes2]
                out_ft[:, :,:, :self.modes2] = torch.einsum("bjix,iox->bjox", a, self.weights1)
        elif self.compression > 0:
            a = x_ft[:, :,:, :self.modes2]
            a = torch.einsum("bjix,ih->bjhx", a, self.weights0)
            a = torch.einsum("bjhx,hkx->bjkx", a, self.weights1)
            out_ft[:, :,:, :self.modes2] = torch.einsum("bjkx,ko->bjox", a, self.weights2)
        # Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x




class FiLMModel(nn.Module):
    """
    Autoformer is the first method to achieve the series-wise connection,
    with inherent O(LlogL) complexity
    """
    def __init__(self, configs, N=512, N2=32):
        super(FiLMModel, self).__init__()
        self.configs = configs
        self.seq_len = configs.seq_len
        # self.label_len = configs.label_len
        # self.pred_len = configs.pred_len
        self.device = configs.device
        self.feat_select = configs.feat_select

        # self.seq_len_all = self.seq_len+self.label_len
        
        # self.output_attention = configs.output_attention
        self.layers = configs.e_layers
        self.modes1 = configs.modes1
        #self.modes1 = 32
        self.enc_in = configs.enc_in
        self.proj = False
        self.e_layers = configs.e_layers
        self.mode_type = configs.mode_type

        if self.configs.ours:
            #b, s, f means b, f
            self.affine_weight = nn.Parameter(torch.ones(1, 1, configs.enc_in))
            self.affine_bias = nn.Parameter(torch.zeros(1, 1, configs.enc_in))
            
        if configs.enc_in > 1000:
            self.proj=True
            self.conv1 = nn.Conv1d(configs.enc_in,configs.d_model,1)
            self.conv2 = nn.Conv1d(configs.d_model,configs.dec_in,1)
            self.d_model = configs.d_model
            self.affine_weight = nn.Parameter(torch.ones(1, 1, configs.d_model))
            self.affine_bias = nn.Parameter(torch.zeros(1, 1, configs.d_model))

        if self.configs.ab == 2:
            self.multiscale = [1, 2, 4]
            #self.multiscale = [1]

            self.window_size=[256]
            self.legts = nn.ModuleList([
                HiPPO_LegT(N=n, dt=1./self.seq_len/i, device=self.device) 
                for n in self.window_size for i in self.multiscale
            ])
            self.spec_conv_1 = nn.ModuleList([
                SpectralConv1d(
                    in_channels=n, 
                    out_channels=n,
                    seq_len=self.seq_len,
                    modes1=configs.modes1,
                    compression=configs.version,
                    ratio=configs.ratio,
                    mode_type=self.mode_type
                ) for n in self.window_size for _ in range(len(self.multiscale))
            ])               
            self.mlp = nn.Linear(len(self.multiscale)*len(self.window_size), 1)
        

        self.fc = nn.Linear(configs.d_model, configs.out_dim)

        if self.feat_select == "fc":
            self.fc_s = nn.Linear(self.seq_len, 1)

    def forward(self, x_enc):

        # decomp init
        
        if self.configs.ab == 2:
            # return_data=[x_enc]

            if self.proj:
                x_enc = self.conv1(x_enc.transpose(1,2))
                x_enc = x_enc.transpose(1,2)

            if self.configs.ours:
                means = x_enc.mean(1, keepdim=True).detach()
                #mean
                x_enc = x_enc - means
                #var
                stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False)+ 1e-5).detach() 
                x_enc /= stdev
                # affine
                x_enc = x_enc * self.affine_weight + self.affine_bias
            # # B, L, E = x_enc.shape
            # # seq_len = self.seq_len

            x_decs = []
            jump_dist=0
            for i in range(len(self.multiscale) * len(self.window_size)):
                x_in_len = self.multiscale[i%len(self.multiscale)] * self.seq_len
                x_in = x_enc[:, -x_in_len:]
                legt = self.legts[i]
                x_in_c = legt(x_in.transpose(1, 2)).permute([1, 2, 3, 0])[:,:,:,jump_dist:]
                out1 = self.spec_conv_1[i](x_in_c)

                x_dec_c = out1.transpose(2, 3)[:,:, self.seq_len-1-jump_dist, :]
                x_dec = x_dec_c @ (legt.eval_matrix[-self.seq_len:,:].T)
                x_decs += [x_dec]
            # return_data.append(x_in_c)
            # return_data.append(out1)

            x_dec = self.mlp(torch.stack(x_decs, dim=-1)).squeeze(-1).permute(0,2,1)
            if self.configs.ours:
                x_dec = x_dec - self.affine_bias
                x_dec = x_dec / (self.affine_weight + 1e-10)
                x_dec = x_dec * stdev
                x_dec = x_dec + means

            if self.proj:
                x_dec = self.conv2(x_dec.transpose(1,2))
                x_dec = x_dec.transpose(1, 2)
        #     # return_data.append(x_dec)
        # if self.output_attention:
        #     return x_dec, return_data
        # else:
        #     return x_dec, None  # [B, L, D]

        feat = x_dec.permute(0, 2, 1) # -> bs, d_model, num_steps
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


class FiLM(nn.Module):

    def __init__(self, params: Namespace):
        super(FiLM, self).__init__()

        seqlen = int(
            (params.max_duration * params.freq / params.downsample) / params.lin_chunk_len
        ) + 1 # +1 for token added during LinearEmbed.

        class Configs(object):
            ab = params.film_ab
            mode_type = params.film_mode_type
            modes1 = params.film_modes1

            # seq_len = 336
            # label_len = 0
            # pred_len = 720
            # output_attention = True
            enc_in = params.film_enc_in
            dec_in = params.film_dec_in
            # embed = 'timeF'
            dropout = 0.05
            # freq = 'h'
            factor = 1
            # n_heads = 8
            # d_ff = 16
            # e_layers = 2
            # d_layers = 1
            # moving_avg = 25
            # c_out = 1
            # wavelet = 0
            ours = False
            version = 16
            ratio = 1

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
        self.backbone = FiLMModel(configs)

    def forward(self, x: Tensor) -> Tensor:

        return self.backbone(x)

