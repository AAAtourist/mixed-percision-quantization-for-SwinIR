import itertools
import torch
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
from .base_model import BaseModel


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
import torch.nn as nn
from basicsr.utils.registry import MODEL_REGISTRY

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


        # start quantization
        x_int = torch.round(x / self.delta) + self.zero_point
        x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
        x_dequant = (x_quant - self.zero_point) * self.delta

        return x_dequant

    '''def init_quantization_scale(self, x: torch.Tensor, channel_wise: bool = False):
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
            best_score = 1e+10
            delta = (x_max - x_min) / (2 ** self.n_bits - 1)
            zero_point = (- x_min / delta).round()
            for pct in [0.9, 0.99, 0.999, 0.9999, 0.99999]:
                try:
                    new_max = torch.quantile(x_clone.reshape(-1), pct)
                    new_min = torch.quantile(x_clone.reshape(-1), 1.0 - pct)
                except:
                    new_max = torch.tensor(np.percentile(
                        x_clone.reshape(-1).cpu(), pct * 100),
                        device=x_clone.device,
                        dtype=torch.float32)
                    new_min = torch.tensor(np.percentile(
                        x_clone.reshape(-1).cpu(), (1 - pct) * 100),
                        device=x_clone.device,
                        dtype=torch.float32)   

                x_q = self.quantize(x_clone, new_max, new_min)
                score = lp_loss(x_clone, x_q, p=2)

                if score < best_score:
                    best_score = score
                    delta = (new_max - new_min) / (2 ** self.n_bits - 1)
                    zero_point = (- new_min / delta).round()

        return delta, zero_point'''

    def quantize(self, x, max, min):
        x_clone = x.clone().detach()    
        delta = (max - min) / (2 ** self.n_bits - 1)
        zero_point = (- min / delta).round()
        x_int = torch.round(x_clone / delta)
        x_quant = torch.clamp(x_int + zero_point, 0, self.n_levels - 1)
        x_float_q = (x_quant - zero_point) * delta
        return x_float_q
    
    def init_quantization_scale(self, max, min):
        self.delta = (max - min) / (2 ** self.n_bits - 1)
        self.zero_point = (- min / self.delta).round()

class Log2Quantizer(nn.Module):
    def __init__(self, n_bits: int = 4, channel_wise: bool = False):
        super(Log2Quantizer, self).__init__()
        assert 2 <= n_bits <= 8, 'bitwidth not supported'
        self.n_bits = n_bits
        self.n_levels = 2 ** self.n_bits
        self.delta = None
        self.inited = False
        self.channel_wise = channel_wise

    def forward(self, x: torch.Tensor):
        sign_x = torch.sign(x)
        abs_x = torch.abs(x) + 0.0001


        # start quantization
        x_dequant = self.quantize(abs_x, self.delta)
        x_dequant *= sign_x
        return x_dequant

    '''
    def init_quantization_scale(self, x: torch.Tensor):
        delta = None
        x_clone = x.clone().detach()
        delta = x_clone.max()
        best_score = 1e+10
        for pct in [0.999, 0.9999, 0.99999]:
            try:
                new_delta = torch.quantile(x_clone.reshape(-1), pct)
            except:
                new_delta = torch.tensor(np.percentile(
                    x_clone.reshape(-1).cpu(), pct * 100),
                    device=x_clone.device,
                    dtype=torch.float32)
            x_q = self.quantize(x_clone, new_delta)
            score = lp_loss(x_clone, x_q, p=2)
            if score < best_score:
                best_score = score
                delta = new_delta

        return delta
    '''
    def quantize(self, x, delta):      
        from math import sqrt
        x_int = torch.round(-1 * (x/delta).log2())
        mask = x_int >= self.n_levels
        x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
        x_float_q = 2**(-1 * torch.ceil(x_quant)) * delta
        x_float_q[mask] = 0
        
        return x_float_q
    
    def init_quantization_scale(self, delta):
        self.delta = delta

def compute_percentile(tensor, percentile):
    '''
        default:channel wise
    '''
    result = tensor
    device = result.device 
    percentile = percentile.to(device)
    if len(tensor.shape) == 3:
        for i in (0, 1):
            result = torch.quantile(result, percentile, dim=i, keepdim=True) # input[1, 1, channels, C]
    elif len(tensor.shape) == 2:
        result = torch.quantile(result, percentile, dim=i, keepdim=True) # weight[1, channels, C]
    else:
        for i in (0, 2, 3):
            result = torch.quantile(result, percentile, dim=i, keepdim=True)
    
    return result

class QuantLinear(nn.Linear):
    def __init__(self,
                 in_features,
                 out_features,
                 input_quantizer=None,
                 weight_quantizer=None):
        super(QuantLinear, self).__init__(in_features, out_features)

        
        self.input_quantizer = input_quantizer
        self.weight_quantizer = weight_quantizer
        self.quant_weight = None

        self.first_time = True

    def forward(self, x):
        if self.first_time:
            global num_linear
            print(f'one linear finish:{num_linear}')
            num_linear += 1
            self.first_time = False
            self.quant_weight = self.weight_quantizer(self.weight)
        
        x = self.input_quantizer(x)
        #w = self.weight_quantizer(self.weight)
        
        out = F.linear(x, weight=self.quant_weight, bias=self.bias)
        #out = F.linear(x, weight=self.weight, bias=self.bias)

        return out

    def search_best_setting(self, origin_output, x):
        if isinstance(self.input_quantizer, UniformQuantizer):
            loss = search_linear_quantizer(self.input_quantizer, self.weight_quantizer, origin_output, x, self.weight, self.bias)
        else:
            loss = search_log2_linear_quantizer(self.input_quantizer, self.weight_quantizer, origin_output, x, self.weight, self.bias)
        
        return loss

class QuantMatMul(nn.Module):
    def __init__(self, 
                 quantizer1=None,
                 quantizer2=None):
        super(QuantMatMul, self).__init__()

        self.quantizer_A = quantizer1
        self.quantizer_B = quantizer2

        self.first_time = True
    
    def forward(self, A, B):
        if self.first_time:
            global num_matmul
            print(f"one matmul finish:{num_matmul}")
            num_matmul += 1
            self.first_time = False

        A = self.quantizer_A(A)
        B = self.quantizer_B(B)
        
        out = A @ B
        return out

    def search_best_setting(self, origin_output, A, B):
        if isinstance(self.quantizer_A, UniformQuantizer):
            loss = search_matmul_quantizer(self.quantizer_A, self.quantizer_B, origin_output, A, B)
        else:
            loss = search_log2_matmul_quantizer(self.quantizer_A, self.quantizer_B, origin_output, A, B)

        return loss

def search_matmul_quantizer(quantizer_A, quantizer_B, origin_output, A, B):
    mat_A = A.clone().detach()
    mat_B = B.clone().detach()

    percentiles = torch.tensor([0.9, 0.99, 0.999, 0.9999, 0.99999])

    combinations = list(itertools.product(percentiles, 1 - percentiles,
                                          percentiles, 1 - percentiles)) 
    combinations = torch.tensor(combinations)
    A_min = combinations[:, 0]
    A_max = combinations[:, 1]
    B_min = combinations[:, 2]
    B_max = combinations[:, 3]

    num_combinations = A_min.size(0)

    A_expanded = mat_A.unsqueeze(-1).expand(-1, -1, -1, -1, num_combinations)
    B_expanded = mat_B.unsqueeze(-1).expand(-1, -1, -1, -1, num_combinations)

    A_min_vals = compute_percentile(mat_A, A_min)
    A_max_vals = compute_percentile(mat_A, A_max)
    B_min_vals = compute_percentile(mat_B, B_min)
    B_max_vals = compute_percentile(mat_B, B_max)

    A_quant = quantizer_A.quantize(A_expanded, A_min_vals, A_max_vals).permute(4, 0, 1, 2, 3)
    B_quant = quantizer_B.quantize(B_expanded, B_min_vals, B_max_vals).permute(4, 0, 1, 2, 3)

    output_quantized = A_quant @ B_quant
    origin_output = origin_output.clone().detach().unsqueeze(0).expand(num_combinations, -1, -1, -1, -1) 

    loss = F.mse_loss(output_quantized, origin_output, reduction='none')  # [C, 2048, 64, 180]
    loss = loss.mean(dim=(1, 2, 3, 4))

    min_loss, best_idx = loss.min(dim=0)

    quantizer_A.init_quantization_scale(A_min_vals.reshape(num_combinations, -1)[best_idx],
                                             A_max_vals.reshape(num_combinations, -1)[best_idx])
    quantizer_B.init_quantization_scale(B_min_vals.reshape(num_combinations, -1)[best_idx],
                                             B_max_vals.reshape(num_combinations, -1)[best_idx])
    
    return min_loss

def search_log2_matmul_quantizer(quantizer_A, quantizer_B, origin_output, A, B):
    mat_A = A.clone().detach()
    mat_B = B.clone().detach()

    percentiles = torch.tensor([0.9, 0.99, 0.999, 0.9999, 0.99999])

    combinations = list(itertools.product(percentiles, percentiles)) 
    combinations = torch.tensor(combinations)
    A_max = combinations[:, 0]
    B_max = combinations[:, 1]

    num_combinations = A_max.size(0)

    A_expanded = mat_A.unsqueeze(-1).expand(-1, -1, -1, -1, num_combinations)
    B_expanded = mat_B.unsqueeze(-1).expand(-1, -1, -1, -1, num_combinations)

    A_max_vals = compute_percentile(mat_A, A_max)
    B_max_vals = compute_percentile(mat_B, B_max)

    A_quant = quantizer_A.quantize(A_expanded, A_max_vals).permute(4, 0, 1, 2, 3)
    B_quant = quantizer_B.quantize(B_expanded, B_max_vals).permute(4, 0, 1, 2, 3)

    output_quantized = A_quant @ B_quant
    origin_output = origin_output.clone().detach().unsqueeze(0).expand(num_combinations, -1, -1, -1, -1) 

    loss = F.mse_loss(output_quantized, origin_output, reduction='none')  # [C, 2048, 64, 180]
    loss = loss.mean(dim=(1, 2, 3, 4))

    min_loss, best_idx = loss.min(dim=0)

    quantizer_A.init_quantization_scale(A_max_vals.reshape(num_combinations, -1)[best_idx])
    quantizer_B.init_quantization_scale(B_max_vals.reshape(num_combinations, -1)[best_idx])
    
    return min_loss

def search_linear_quantizer(input_quantizer, weight_quantizer, origin_output, x, weight, bias):
    input_tensor = x.clone().detach()
    weight_tensor = weight.clone().detach()

    percentiles = torch.tensor([0.9, 0.99, 0.999, 0.9999, 0.99999])

    combinations = list(itertools.product(percentiles, 1 - percentiles,
                                          percentiles, 1 - percentiles)) 
    combinations = torch.tensor(combinations)
    input_min = combinations[:, 0]
    input_max = combinations[:, 1]
    weight_min = combinations[:, 2]
    weight_max = combinations[:, 3]

    num_combinations = input_min.size(0)
    b, n, _ = input_tensor.shape
    out_c = weight_tensor.shape[0]

    input_expanded = input_tensor.unsqueeze(-1).expand(-1, -1, -1, num_combinations)  # [2048, 64, 60, C]
    weight_expanded = weight_tensor.unsqueeze(-1).expand(-1, -1, num_combinations)  # [180, 60, C]

    input_min_vals = compute_percentile(input_tensor, input_min)  # [1, 1, channels, C]
    input_max_vals = compute_percentile(input_tensor, input_max)  # [1, 1, channels, C]
    weight_min_vals = compute_percentile(weight_tensor, weight_min)  # [1, channels, C]
    weight_max_vals = compute_percentile(weight_tensor, weight_max)  # [1, channels, C]

    input_quant = input_quantizer.quantize(input_expanded, input_min_vals, input_max_vals) # [2048, 64, 60, C]
    weight_quant = weight_quantizer.quantize(weight_expanded, weight_min_vals, weight_max_vals)  # [180, 60, C]

    input_quant = input_quant.view(-1, input_tensor.size(2), num_combinations) #[2048, 64, 60, C] -> [2048 * 64, 60, C]
    weight_quant = weight_quant.permute(2, 1, 0)  # [C, 60, 180]

    output_quantized = torch.bmm(input_quant.transpose(0, 2), weight_quant)
    output_quantized = output_quantized.view(num_combinations, b, n, out_c) + bias  # [C, 2048, 64, 180]

    origin_output = origin_output.clone().detach().unsqueeze(0).expand(num_combinations, -1, -1, -1) 

    loss = F.mse_loss(output_quantized, origin_output, reduction='none')  # [C, 2048, 64, 180]
    loss = loss.mean(dim=(1, 2, 3))

    min_loss, best_idx = loss.min(dim=0)

    #weight_quantizer.weight.data = weight_quant[best_idx].transpose(0, 1)
    weight_quantizer.init_quantization_scale(weight_min_vals.reshape(num_combinations, -1)[best_idx],
                                             weight_max_vals.reshape(num_combinations, -1)[best_idx])
    input_quantizer.init_quantization_scale(input_min_vals.reshape(num_combinations, -1)[best_idx],
                                             input_max_vals.reshape(num_combinations, -1)[best_idx])

    return min_loss

def search_log2_linear_quantizer(input_quantizer, weight_quantizer, origin_output, x, weight, bias):
    input_tensor = x.clone().detach()
    weight_tensor = weight.clone().detach()

    percentiles = torch.tensor([0.9, 0.99, 0.999, 0.9999, 0.99999])

    combinations = list(itertools.product(percentiles, percentiles)) 
    combinations = torch.tensor(combinations)

    input_delta = combinations[:, 0]
    weight_delta = combinations[:, 1]

    num_combinations = input_delta.size(0)
    b, n, _ = input_tensor.shape
    out_c = weight_tensor.shape[0]

    input_expanded = input_tensor.unsqueeze(-1).expand(-1, -1, -1, num_combinations)  # [2048, 64, 60, C]
    weight_expanded = weight_tensor.unsqueeze(-1).expand(-1, -1, num_combinations)  # [180, 60, C]

    input_max_vals = compute_percentile(input_tensor, input_delta)  # [1, 1, channels, C]
    weight_max_vals = compute_percentile(weight_tensor, weight_delta)  # [1, channels, C]

    input_quant = input_quantizer.quantize(input_expanded, input_max_vals) # [2048, 64, 60, C]
    weight_quant = weight_quantizer.quantize(weight_expanded, weight_max_vals)  # [180, 60, C]

    input_quant = input_quant.view(-1, input_tensor.size(2), num_combinations) #[2048, 64, 60, C] -> [2048 * 64, 60, C]
    weight_quant = weight_quant.permute(2, 1, 0)  # [C, 60, 180]

    output_quantized = torch.bmm(input_quant.transpose(0, 2), weight_quant)
    output_quantized = output_quantized.view(num_combinations, b, n, out_c) + bias  # [C, 2048, 64, 180]

    origin_output = origin_output.clone().detach().unsqueeze(0).expand(num_combinations, -1, -1, -1) 

    loss = F.mse_loss(output_quantized, origin_output, reduction='none')  # [C, 2048, 64, 180]
    loss = loss.mean(dim=(1, 2, 3))

    min_loss, best_idx = loss.min(dim=0)

    input_quantizer.init_quantization_scale(input_max_vals.reshape(num_combinations, -1)[best_idx])
    weight_quantizer.init_quantization_scale(weight_max_vals.reshape(num_combinations, -1)[best_idx])

    return min_loss

def create_quantizers(default_quant_params={}):
    quantizers = []
    for i in [2, 3, 4, 8]:
        #quant_param = {**default_quant_params, "n_bits": i}
        input_param = {**default_quant_params["input_quant_params"], "n_bits": i}
        weight_param = {**default_quant_params["weight_quant_params"], "n_bits": i}
        #log_quant_param = {**default_quant_params, "n_bits": i, "log_quant": True}

        uni_quantizer1 = UniformQuantizer(**input_param)
        uni_quantizer2 = UniformQuantizer(**weight_param)

        log_quantizer1 = Log2Quantizer(**input_param)
        log_quantizer2 = Log2Quantizer(**weight_param)

        uni_quantizer_name = f"uni_quantizer_bit{i}"
        log_quantizer_name = f"log_quantizer_bit{i}"

        quantizers.append({
            "name": uni_quantizer_name,
            "quantizer1": uni_quantizer1,
            "quantizer2": uni_quantizer2
        })
        quantizers.append({
            "name": log_quantizer_name,
            "quantizer1": log_quantizer1,
            "quantizer2": log_quantizer2
        })
    
    return quantizers

class List_Quantizers(nn.Module):
    def __init__(self):
        super(List_Quantizers, self).__init__()

        self.quantizers_dict = {}
        self.need_search = True

    def forward(self, *args):

        if self.need_search:
            origin_output = self.quantizers_dict["origin_module"](*args)
            self.module_search(origin_output, *args)
            self.need_search = False

        return self.quantizers_dict["best_module"](*args)

    def append_quantizer(self, name, module):
        self.quantizers_dict[name] = module

    def module_search(self, origin_output, *args):
        best_score = 1e+10
        #print(self.quantizers_dict.items())
        for name, module in self.quantizers_dict.items():
            if name == "origin_module":
                continue
            loss = module.search_best_setting(origin_output, *args)
            if loss < best_score:
                best_score = loss
                self.quantizers_dict["best_module"] = module

def quant_model(model, quant_params={}):

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

            quantizers = create_quantizers(quant_params)

            linear_quantizers = List_Quantizers()

            linear_quantizers.append_quantizer("origin_module", m)
            for quantizer_info in quantizers:
                input_quantizer = quantizer_info["quantizer1"]
                weight_quantizer = quantizer_info["quantizer2"]

                new_m = QuantLinear(m.in_features, m.out_features, input_quantizer, weight_quantizer)
                new_m.weight.data = m.weight.data
                new_m.bias = m.bias

                linear_quantizers.append_quantizer(quantizer_info["name"], new_m)

            setattr(father_module, name[idx:], linear_quantizers)
        elif isinstance(m, MatMul):
            # Matmul Layer
            idx = idx + 1 if idx != 0 else idx

            quantizers = create_quantizers(quant_params)

            matmul_quantizers = List_Quantizers()

            matmul_quantizers.append_quantizer("origin_module", m)
            for quantizer_info in quantizers:
                quantizer1 = quantizer_info["quantizer1"]
                quantizer2 = quantizer_info["quantizer2"]

                new_m = QuantMatMul(quantizer1, quantizer2)

                matmul_quantizers.append_quantizer(quantizer_info["name"], new_m)

            setattr(father_module, name[idx:], matmul_quantizers)

    return model

@MODEL_REGISTRY.register()
class SRModel(BaseModel):
    """Base SR model for single image super-resolution."""

    def __init__(self, opt):
        super(SRModel, self).__init__(opt)

        # define network
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        #self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)

        if 'quantization' in self.opt:
            self.net_g = quant_model(
                model = self.net_g,
                quant_params=self.opt['quantization']
                )
            self.net_g = self.model_to_device(self.net_g)
            self.cali_data = torch.load(opt['cali_data'])
            with torch.no_grad():
                print('Performing initial quantization ...')
                self.feed_data(self.cali_data)
                _ = self.net_g(self.lq)
                print('initial quantization over ...')

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if self.cri_pix is None and self.cri_perceptual is None:
            raise ValueError('Both pixel and perceptual losses are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix
        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def test(self):
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(self.lq)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(self.lq)
            self.net_g.train()

    def test_selfensemble(self):
        # TODO: to be tested
        # 8 augmentations
        # modified from https://github.com/thstkdgus35/EDSR-PyTorch

        def _transform(v, op):
            # if self.precision != 'single': v = v.float()
            v2np = v.data.cpu().numpy()
            if op == 'v':
                tfnp = v2np[:, :, :, ::-1].copy()
            elif op == 'h':
                tfnp = v2np[:, :, ::-1, :].copy()
            elif op == 't':
                tfnp = v2np.transpose((0, 1, 3, 2)).copy()

            ret = torch.Tensor(tfnp).to(self.device)
            # if self.precision == 'half': ret = ret.half()

            return ret

        # prepare augmented data
        lq_list = [self.lq]
        for tf in 'v', 'h', 't':
            lq_list.extend([_transform(t, tf) for t in lq_list])

        # inference
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                out_list = [self.net_g_ema(aug) for aug in lq_list]
        else:
            self.net_g.eval()
            with torch.no_grad():
                out_list = [self.net_g_ema(aug) for aug in lq_list]
            self.net_g.train()

        # merge results
        for i in range(len(out_list)):
            if i > 3:
                out_list[i] = _transform(out_list[i], 't')
            if i % 4 > 1:
                out_list[i] = _transform(out_list[i], 'h')
            if (i % 4) % 2 == 1:
                out_list[i] = _transform(out_list[i], 'v')
        output = torch.cat(out_list, dim=0)

        self.output = output.mean(dim=0, keepdim=True)

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']])
            metric_data['img'] = sr_img
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']])
                metric_data['img2'] = gt_img
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["name"]}.png')
                imwrite(sr_img, save_img_path)

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    self.metric_results[name] += calculate_metric(metric_data, opt_)
            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')
        if use_pbar:
            pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)
