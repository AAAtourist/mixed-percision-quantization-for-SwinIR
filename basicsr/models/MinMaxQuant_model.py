from functools import partial
import os
from re import L
import time

import numpy as np
import torch
from collections import OrderedDict
from copy import deepcopy
from torch.nn.parallel import DataParallel, DistributedDataParallel
import torch.optim

from basicsr.models import lr_scheduler as lr_scheduler
from basicsr.utils.dist_util import master_only
from basicsr.archs import build_network
from basicsr.losses import build_loss
import torch.nn as nn
from torch import Tensor
from basicsr.utils.registry import MODEL_REGISTRY
from torch.optim import Adam

from basicsr.archs.swinir_arch import SwinTransformerBlock, WindowAttention, MatMul
from basicsr.utils import get_root_logger, imwrite, tensor2img
from os import path as osp
from basicsr.metrics import calculate_metric

from torchvision.transforms import Resize
from torch.nn import functional as F
from types import MethodType
from tqdm import tqdm
import torch.multiprocessing as mp
from torch.nn import Parameter

import warnings

num_linear = 0
num_matmul = 0

def lp_loss(pred, tgt, p=2.0):
    return (pred-tgt).abs().pow(p).mean()

class UniformQuantizer(nn.Module):
    def __init__(self, n_bits: int = 8, channel_wise: bool = False):
        super(UniformQuantizer, self).__init__()
        assert 2 <= n_bits <= 8, 'bitwidth not supported'
        self.n_bits = n_bits
        self.n_levels = 2 ** self.n_bits
        self.delta = None
        self.zero_point = None
        self.inited = False
        self.channel_wise = channel_wise

    def forward(self, x: torch.Tensor):

        if self.inited is False:
            self.delta, self.zero_point = self.init_quantization_scale(x, self.channel_wise)
            self.inited = True

        # start quantization
        x_int = torch.round(x / self.delta) + self.zero_point
        x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
        x_dequant = (x_quant - self.zero_point) * self.delta

        return x_dequant

    def init_quantization_scale(self, x: torch.Tensor, channel_wise: bool = False):
        delta, zero_point = None, None
        if channel_wise:
            x_clone = x.clone().detach()
            n_channels = x_clone.shape[-1] if len(x.shape) == 3 else x_clone.shape[0]
            if len(x.shape) == 4:
                n_channels = x_clone.shape[1] * x_clone.shape[-1]
                x_max = x_clone.abs().max(dim=0)[0].max(dim=1)[0].reshape(-1)
                #x_max = x_clone.abs().max(dim=-1)[0].max(dim=-1)[0].max(dim=-1)[0]
            elif len(x.shape) == 2:
                x_max = x_clone.abs().max(dim=-1)[0]
            elif len(x.shape) == 3:
                x_max = x_clone.abs().max(dim=0)[0].max(dim=0)[0]
            else:
                raise NotImplementedError

            delta = x_max.clone()
            zero_point = x_max.clone()
            # determine the scale and zero point channel-by-channel
            for c in range(n_channels):
                if len(x.shape) == 3:
                    delta[c], zero_point[c] = self.init_quantization_scale(x_clone[:,:,c], channel_wise=False)
                elif len(x.shape) == 4:
                    delta[c], zero_point[c] = self.init_quantization_scale(x_clone[:, c // x_clone.shape[-1], :, c // x_clone.shape[1]], channel_wise=False)
                else:
                    delta[c], zero_point[c] = self.init_quantization_scale(x_clone[c], channel_wise=False)
            if len(x.shape) == 4:
                delta = delta.view(1, x_clone.shape[1], 1, -1)
                zero_point = zero_point.view(1, x_clone.shape[1], 1, -1)
            elif len(x.shape) == 2:
                delta = delta.view(-1, 1)
                zero_point = zero_point.view(-1, 1)
            elif len(x.shape) == 3:
                delta = delta.view(1, 1, -1)
                zero_point = zero_point.view(1, 1, -1)
            else:
                raise NotImplementedError
        else:
            x_clone = x.clone().detach()
            x_max = x_clone.max()
            x_min = x_clone.min()
            best_score = lp_loss(x_clone, x_max, x_min)
            delta = (x_max - x_min) / (2 ** self.n_bits - 1)
            zero_point = (- x_min / delta).round()
            for pct in [0.9, 0.99, 0.999]:
                new_max = torch.quantile(x_clone.reshape(-1), pct)
                new_min = torch.quantile(x_clone.reshape(-1), 1.0 - pct)

                x_q = self.quantize(x_clone, new_max, new_min)
                score = lp_loss(x_clone, x_q, p=2)

                if score < best_score:
                    best_score = score
                    delta = (new_max - new_min) / (2 ** self.n_bits - 1)
                    zero_point = (- new_min / delta).round()

        return delta, zero_point

    def quantize(self, x, max, min):
        delta = (max - min) / (2 ** self.n_bits - 1)
        zero_point = (- min / delta).round()
        x_int = torch.round(x / delta)
        x_quant = torch.clamp(x_int + zero_point, 0, self.n_levels - 1)
        x_float_q = (x_quant - zero_point) * delta
        return x_float_q


class QuantLinear(nn.Linear):
    def __init__(self,
                 in_features,
                 out_features,
                 input_quant_params={},
                 weight_quant_params={}):
        super(QuantLinear, self).__init__(in_features, out_features)

        self.input_quantizer = UniformQuantizer(**input_quant_params)
        self.weight_quantizer = UniformQuantizer(**weight_quant_params)

        self.use_input_quant = True
        self.use_weight_quant = True
        self.first_time = True

    def forward(self, x):
        if self.first_time:
            global num_linear
            print(f'one linear finish:{num_linear}')
            num_linear += 1
            self.first_time = False
        if self.use_input_quant:
            x = self.input_quantizer(x)

        if self.use_weight_quant:
            w = self.weight_quantizer(self.weight)
        else:
            w = self.weight

        out = F.linear(x, weight=w, bias=self.bias)

        return out

class QuantMatMul(nn.Module):
    def __init__(self,
                 input_quant_params={}):
        super(QuantMatMul, self).__init__()

        input_quant_params_matmul = deepcopy(input_quant_params)

        self.quantizer_A = UniformQuantizer(**input_quant_params_matmul)
        self.quantizer_B = UniformQuantizer(**input_quant_params_matmul)

        self.use_input_quant = True
        self.first_time = True
    
    def forward(self, A, B):
        if self.first_time:
            global num_matmul
            print(f"one matmul finish:{num_matmul}")
            num_matmul += 1
            self.first_time = False
        if self.use_input_quant:
            A = self.quantizer_A(A)
            B = self.quantizer_B(B)
        
        out = A @ B
        return out


def MinMaxQuant(model, input_quant_params={}, weight_quant_params={}):

    module_dict={}
    for name, m in model.named_modules():
        module_dict[name] = m
        idx = name.rfind('.')
        if idx == -1:
            idx = 0
        father_name = name[:idx]
        if father_name in module_dict:
            father_module = module_dict[father_name]
        else:
            raise RuntimeError(f"father module {father_name} not found")

        if isinstance(m, nn.Linear):
            # Linear Layer
            idx = idx + 1 if idx != 0 else idx
            new_m = QuantLinear(m.in_features, m.out_features, input_quant_params, weight_quant_params)
            new_m.weight.data = m.weight.data
            new_m.bias = m.bias
            setattr(father_module, name[idx:], new_m)
        elif isinstance(m, MatMul):
            # Matmul Layer
            idx = idx + 1 if idx != 0 else idx
            new_m = QuantMatMul(input_quant_params)
            setattr(father_module, name[idx:], new_m)

    return model