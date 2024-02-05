from argparse import Namespace
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from codes.architectures.transformer import LinearEmbed, Transformer

def prepare_model(params: Namespace) -> nn.Module:
    """
    Args:
        params (Namespace):
    Returns:
        predictor (nn.Module):
    """
    # Transformer
    if params.modelname == "transformer":
        foot = LinearEmbed(params)
        backbone = Transformer(params)

    elif params.modelname == "fedformer":
        from codes.architectures.fedformer import FEDformer
        foot = LinearEmbed(params)
        backbone = FEDformer(params)

    elif params.modelname == "informer":
        from codes.architectures.informer import Informer
        foot = LinearEmbed(params)
        backbone = Informer(params)

    elif params.modelname == "autoformer":
        from codes.architectures.autoformer import Autoformer
        foot = LinearEmbed(params)
        backbone = Autoformer(params)

    elif params.modelname == "film":
        from codes.architectures.film import FiLM
        foot = LinearEmbed(params)
        backbone = FiLM(params)

    elif params.modelname == "luna":
        from codes.architectures.luna import LunaTransformer
        foot = LinearEmbed(params)
        backbone = LunaTransformer(params)

    elif params.modelname == "nystrom":
        from codes.architectures.nystromformer import Nystromformer
        foot = LinearEmbed(params)
        backbone = Nystromformer(params)
    
    elif params.modelname == "lintrans":
        from codes.architectures.linear_transformer import LinearTransformer
        foot = LinearEmbed(params)
        backbone = LinearTransformer(params)

    elif params.modelname == "performer":
        from codes.architectures.performer import Performer
        foot = LinearEmbed(params)
        backbone = Performer(params)

    # ResNet18
    elif params.modelname == "resnet18":
        from codes.architectures.resnet import ResNet18
        foot = None
        backbone = ResNet18(params)    

    # ResNet34
    elif params.modelname == "resnet34":
        from codes.architectures.resnet import ResNet34
        foot = None
        backbone = ResNet34(params)

    # ResNet50
    elif params.modelname == "resnet50":
        from codes.architectures.resnet import ResNet50
        foot = None
        backbone = ResNet50(params)

    # EfficientNet-B0
    elif params.modelname == "effnetb0":
        from codes.architectures.efficient_net import effnet1d_b0
        foot = None

        seqlen = int(
            (params.max_duration * params.freq / params.downsample)
        )
        effnet_params = {
            "num_lead": params.num_lead,
            "sequence_length": seqlen,
            "backbone_out_dim": params.backbone_out_dim
        }
        backbone = effnet1d_b0(**effnet_params)

    # LSTM
    elif params.modelname == "lstm":
        from codes.architectures.bi_lstm import VarDepthLSTM
        foot = None
        backbone = VarDepthLSTM(params, params.num_lead)

    elif params.modelname == "emblstm":
        from codes.architectures.bi_lstm import VarDepthLSTM
        foot = LinearEmbed(params, add_cls_token=False)
        backbone = VarDepthLSTM(params,  params.emb_dim)

    elif params.modelname == "gru":
        from codes.architectures.bi_gru import VarDepthGRU
        foot = None
        backbone = VarDepthGRU(params, params.num_lead)

    elif params.modelname == "embgru":
        from codes.architectures.bi_gru import VarDepthGRU
        foot = LinearEmbed(params, add_cls_token=False)
        backbone = VarDepthGRU(params, params.emb_dim)

    elif params.modelname == "s4":
        from codes.architectures.s4 import S4
        foot = LinearEmbed(params)
        backbone = S4(params)

    elif params.modelname == "mega":
        from codes.architectures.mega import Mega
        foot = LinearEmbed(params)
        backbone = Mega(params)

    else:
        raise NotImplementedError(f"{params.modelname} is not implemented.")

    head = HeadModule(params.backbone_out_dim)
    model = Predictor(backbone, head, foot)
    return model    

class Predictor(nn.Module):

    def __init__(
        self, 
        backbone: nn.Module, 
        head: nn.Module,
        foot: Optional[nn.Module]
    ) -> None:
        super(Predictor, self).__init__()

        self.backbone = backbone
        self.head = head
        self.foot = foot

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (torch.Tensor): Tensor of size 
                (batch_size, num_lead, seq_len).
        Returns:
            h (torch.Tensor): Tensor of size (batch_size, num_classes)
        """
        if self.foot is not None:
            x = self.foot(x)
        h = self.backbone(x) # (bs, params.backbone_out_dim)
        h = self.head(h) 
        return h

class HeadModule(nn.Module):

    def __init__(self, in_dim: int):
        super(HeadModule, self).__init__()

        self.fc1 = nn.Linear(in_dim, 32)
        self.drop1 = nn.Dropout(0.25)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (torch.Tensor): Tensor of size (num_batch, in_dim).
        Returns:
            feat (torch.Tensor): Tensor of size (num_batch, 1).
        """
        feat = F.relu(self.fc1(x))
        feat = self.drop1(feat)
        feat = self.fc2(feat)
        return feat