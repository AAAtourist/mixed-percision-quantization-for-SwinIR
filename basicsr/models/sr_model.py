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
        self.delta = self.delta.to(x.device)
        self.zero_point = self.zero_point.to(x.device)

        x_int = torch.round(x / self.delta) + self.zero_point
        x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
        x_dequant = (x_quant - self.zero_point) * self.delta

        return x_dequant

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

        # start quantization
        x_dequant = self.quantize(x, self.delta)
        return x_dequant

    def quantize(self, x, delta):
        epsilon = 1e-8
        x = x.to(delta.device)
        sign_x = torch.sign(x)
        abs_x = torch.abs(x)      
        safe_abs_x = torch.clamp(abs_x, min=epsilon)
        x_int = torch.round(-1 * (safe_abs_x/delta).log2())
        mask = x_int >= self.n_levels
        x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
        x_float_q = 2**(-1 * torch.ceil(x_quant)) * delta
        x_float_q[mask] = 0
        x_float_q *= sign_x
        return x_float_q
    
    def init_quantization_scale(self, delta):
        self.delta = delta

def channel_percentile(x, pct, channel_wise=True):
    x_clone = x.clone().detach()
    
    if channel_wise:
        num = x.shape[-1] if len(x.shape) != 4 else x.shape[-1] * x.shape[1]
        max_per = torch.zeros(num)
        for channel in range(num):
            if len(x.shape) == 3:
                max_per[channel] = channel_percentile(x_clone[:,:, channel],
                                                       pct, channel_wise=False)
            elif len(x.shape) == 4:
                max_per[channel] = channel_percentile(x_clone[:,channel // x.shape[-1],:, channel % x.shape[-1]],
                                                       pct, channel_wise=False)
            else:
                max_per[channel] = channel_percentile(x_clone[:, channel],
                                                       pct, channel_wise=False)
        if len(x.shape) == 4:
            pct_tensor = max_per.view(1, x.shape[1], 1, -1)
        elif len(x.shape) == 3:
            pct_tensor = max_per.view(1, 1, -1)
        else:
            pct_tensor = max_per.view(1, -1)
    else:
        try:
            pct_tensor = torch.quantile(x_clone.reshape(-1), pct)
        except:
            pct_tensor = torch.tensor(np.percentile(
                x_clone.reshape(-1).cpu(), pct * 100),
                device=x_clone.device,
                dtype=torch.float32)
    return pct_tensor.cuda()

def compute_percentile(x, pct, log=False):
    num_com = x.shape[-1]
    if log:
        x = torch.abs(x)
    return torch.stack([channel_percentile(x[..., com], pct[com], channel_wise=True) for com in range(num_com)], dim=-1)

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
        print('start linear search')
        if isinstance(self.input_quantizer, UniformQuantizer):
            print('search uni')
            loss = search_linear_quantizer(self.input_quantizer, self.weight_quantizer, origin_output, x, self.weight, self.bias)
        else:
            print('search log')
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
        print('start matmul search')
        if isinstance(self.quantizer_A, UniformQuantizer):
            loss = search_matmul_quantizer(self.quantizer_A, self.quantizer_B, origin_output, A, B)
        else:
            loss = search_log2_matmul_quantizer(self.quantizer_A, self.quantizer_B, origin_output, A, B)

        return loss

def search_matmul_quantizer(quantizer_A, quantizer_B, origin_output, A, B):
    mat_A = A.clone().detach()
    mat_B = B.clone().detach()

    percentiles = torch.tensor([0.9, 0.99, 0.999, 0.9999])

    combinations = list(itertools.product(percentiles, 1 - percentiles))
    combinations = torch.tensor(combinations).to(mat_A.device)
    A_min = combinations[:, 1]
    A_max = combinations[:, 0]
    B_min = combinations[:, 1]
    B_max = combinations[:, 0]
    num_combinations = A_min.size(0)    

    A_expanded = mat_A.unsqueeze(-1).expand(-1, -1, -1, -1, num_combinations)
    B_expanded = mat_B.unsqueeze(-1).expand(-1, -1, -1, -1, num_combinations)

    A_min_vals = compute_percentile(A_expanded, A_min)#torch.Size([1, 6, 1, 10, 16])
    A_max_vals = compute_percentile(A_expanded, A_max)
    B_min_vals = compute_percentile(B_expanded, B_min)
    B_max_vals = compute_percentile(B_expanded, B_max)

    A_quant = quantizer_A.quantize(A_expanded, A_min_vals, A_max_vals)
    B_quant = quantizer_B.quantize(B_expanded, B_min_vals, B_max_vals)
    A_quant = A_quant.permute(4, 0, 1, 2, 3)
    B_quant = B_quant.permute(4, 0, 1, 2, 3)[None, ...]
    origin_batch = origin_output[None, None, ...]

    best_score = 1e+10
    best_idx1, best_idx2 = 0, 0
    batch = len(percentiles)

    for i in range(0, num_combinations, batch):
        #A_batch = A_quant[i][None, ...]
        end_idx = min(i + batch, num_combinations)
        A_batch = torch.stack([A_quant[idx][None, ...] for idx in range(i, end_idx)], dim=0) #[batch, 1, ...]
        with torch.no_grad():
            output_batch = A_batch @ B_quant
        del A_batch
        loss = F.mse_loss(output_batch, origin_batch, reduction='none')
        loss = loss.mean(dim=(2, 3, 4, 5))

        min_mse_idx = torch.argmin(loss)
        idx1 = min_mse_idx // num_combinations
        idx2 = min_mse_idx % num_combinations
        min_loss = loss[idx1, idx2]

        if min_loss < best_score:
            best_score = min_loss
            best_idx1 = idx1 + i
            best_idx2 = idx2
        del output_batch, loss
        torch.cuda.empty_cache()

    quantizer_A.init_quantization_scale(A_min_vals[..., best_idx1],
                                             A_max_vals[..., best_idx1])
    quantizer_B.init_quantization_scale(B_min_vals[..., best_idx2],
                                             B_max_vals[..., best_idx2])
    '''
        index = torch.arange(num_combinations).unsqueeze(-1).repeat(1, num_combinations).reshape(-1)
        A_min_vals = A_min_vals[..., index]
        A_max_vals = A_max_vals[..., index]
        A_quant = A_quant.cpu()[..., index]
        print('check5')
        B_quant = B_quant.cpu().unsqueeze(-2).expand(-1, -1, -1, -1, num_combinations, num_combinations)
        B_quant = B_quant.reshape(*B_quant.shape[:-2], -1).permute(4, 0, 1, 2, 3)
        B_min_vals = B_min_vals.unsqueeze(-2).expand(-1, -1, -1, -1, num_combinations, num_combinations).reshape(*B_min_vals.shape[:-1], -1)
        B_max_vals = B_max_vals.unsqueeze(-2).expand(-1, -1, -1, -1, num_combinations, num_combinations).reshape(*B_max_vals.shape[:-1], -1)
        num_combinations *= num_combinations
        #A_quant = transfer_to_cuda(A_quant, num_combinations)
        #B_quant = transfer_to_cuda(B_quant, num_combinations)
        print('check6')
        batch_size = 64

        for i in range(0, num_combinations, batch_size):
            end_idx = min(i + batch_size, num_combinations)
            print('check7')
            A_batch = A_quant[i:end_idx, ...].cuda(non_blocking=True)
            B_batch = B_quant[i:end_idx, ...].cuda(non_blocking=True)
            print('ckeck7.5')
            with torch.no_grad():
                output_batch = A_batch @ B_batch
            #output_batch = output_batch.view((end_idx - i), b, n, out_c).cuda() + bias
            origin_batch = origin_output.clone().detach().unsqueeze(0).expand((end_idx - i), -1, -1, -1, -1)
            print('check8')
            print(output_batch.device)
            print(origin_batch.device)
            del A_batch, B_batch
            loss = F.mse_loss(output_batch, origin_batch, reduction='none')
            loss = loss.mean(dim=(1, 2, 3, 4))
            print(loss.is_cuda)

            min_loss, idx = loss.min(dim=0)
            if min_loss < best_score:
                best_score = min_loss
                best_idx = idx

            del output_batch, loss, origin_batch
            torch.cuda.empty_cache()

        quantizer_A.init_quantization_scale(A_min_vals.reshape(num_combinations, -1)[best_idx],
                                                A_max_vals.reshape(num_combinations, -1)[best_idx])
        quantizer_B.init_quantization_scale(B_min_vals.reshape(num_combinations, -1)[best_idx],
                                                B_max_vals.reshape(num_combinations, -1)[best_idx])
    '''
    
    return best_score

def search_log2_matmul_quantizer(quantizer_A, quantizer_B, origin_output, A, B):
    mat_A = A.clone().detach()
    mat_B = B.clone().detach()

    percentiles = torch.tensor([0.9, 0.99, 0.999, 0.9999])

    combinations = list(itertools.product(percentiles, percentiles)) 
    combinations = torch.tensor(combinations).to(mat_A.device)
    A_max = combinations[:, 0]
    B_max = combinations[:, 1]

    num_combinations = A_max.size(0)

    A_expanded = mat_A.unsqueeze(-1).expand(-1, -1, -1, -1, num_combinations)
    B_expanded = mat_B.unsqueeze(-1).expand(-1, -1, -1, -1, num_combinations)

    A_max_vals = compute_percentile(A_expanded, A_max, log=True)
    B_max_vals = compute_percentile(B_expanded, B_max, log=True)

    A_quant = quantizer_A.quantize(A_expanded, A_max_vals)
    B_quant = quantizer_B.quantize(B_expanded, B_max_vals)
    A_quant = A_quant.permute(4, 0, 1, 2, 3)
    B_quant = B_quant.permute(4, 0, 1, 2, 3)[None, ...]
    origin_batch = origin_output[None, None, ...]

    best_score = 1e+10
    best_idx1, best_idx2 = 0, 0
    batch = len(percentiles)

    for i in range(0, num_combinations, batch):
        #A_batch = A_quant[i][None, ...]
        end_idx = min(i + batch, num_combinations)
        A_batch = torch.stack([A_quant[idx][None, ...] for idx in range(i, end_idx)], dim=0) #[batch, 1, ...]
        with torch.no_grad():
            output_batch = A_batch @ B_quant
        del A_batch
        loss = F.mse_loss(output_batch, origin_batch, reduction='none')
        loss = loss.mean(dim=(2, 3, 4, 5))

        min_mse_idx = torch.argmin(loss)
        idx1 = min_mse_idx // num_combinations
        idx2 = min_mse_idx % num_combinations
        min_loss = loss[idx1, idx2]

        if min_loss < best_score:
            best_score = min_loss
            best_idx1 = idx1 + i
            best_idx2 = idx2
        del output_batch, loss
        torch.cuda.empty_cache()

    quantizer_A.init_quantization_scale(A_max_vals[..., best_idx1])
    quantizer_B.init_quantization_scale(B_max_vals[..., best_idx2])
    '''
        index = torch.arange(num_combinations).unsqueeze(-1).repeat(1, num_combinations).reshape(-1)
        A_quant = A_quant.cpu()[..., index]
        A_quant = A_quant.permute(4, 0, 1, 2, 3)
        #A_quant = transfer_to_cuda(A_quant, num_combinations)
        B_quant = B_quant.cpu().unsqueeze(-2).expand(-1, -1, -1, -1, num_combinations, num_combinations)
        B_quant = B_quant.reshape(*B_quant.shape[:-2], -1).permute(4, 0, 1, 2, 3)
        #B_quant = transfer_to_cuda(B_quant, num_combinations)
        A_max_vals = A_max_vals[..., index]
        B_max_vals = B_max_vals.unsqueeze(-2).expand(-1, -1, -1, -1, num_combinations, num_combinations).reshape(*B_max_vals.shape[:-1], -1)

        batch_size = 64
        best_score = 1e+10
        best_idx = 0

        for i in range(0, num_combinations, batch_size):
            end_idx = min(i + batch_size, num_combinations)
            A_batch = A_quant[i:end_idx, ...].cuda(non_blocking=True)
            B_batch = B_quant[i:end_idx, ...].cuda(non_blocking=True)

            with torch.no_grad():
                output_batch = A_batch @ B_batch
            origin_batch = origin_output.clone().detach().unsqueeze(0).expand((end_idx - i), -1, -1, -1, -1)

            loss = F.mse_loss(output_batch, origin_batch, reduction='none')
            loss = loss.mean(dim=(1, 2, 3, 4))
            print(loss.is_cuda)

            min_loss, idx = loss.min(dim=0)
            if min_loss < best_score:
                best_score = min_loss
                best_idx = idx

            del A_batch, B_batch, output_batch, loss, origin_batch
            torch.cuda.empty_cache()

        quantizer_A.init_quantization_scale(A_max_vals.reshape(num_combinations, -1)[best_idx])
        quantizer_B.init_quantization_scale(B_max_vals.reshape(num_combinations, -1)[best_idx])
    '''
    return best_score

def search_linear_quantizer(input_quantizer, weight_quantizer, origin_output, x, weight, bias):
    input_tensor = x.clone().detach()
    weight_tensor = weight.clone().detach()

    percentiles = torch.tensor([0.99, 0.999, 0.9999, 0.99999])

    combinations = list(itertools.product(percentiles, 1 - percentiles)) 
    combinations = torch.tensor(combinations)
    input_min = combinations[:, 1]
    input_max = combinations[:, 0]
    weight_min = combinations[:, 1]
    weight_max = combinations[:, 0]

    num_combinations = input_min.size(0)

    input_expanded = input_tensor.unsqueeze(-1).expand(-1, -1, -1, num_combinations)  # [2048, 64, 60, C]
    weight_expanded = weight_tensor.unsqueeze(-1).expand(-1, -1, num_combinations)  # [180, 60, C]
    
    input_min_vals = compute_percentile(input_expanded, input_min)  # [1, 1, channels, C]
    input_max_vals = compute_percentile(input_expanded, input_max)  # [1, 1, channels, C]
    weight_min_vals = compute_percentile(weight_expanded, weight_min)  # [1, channels, C]
    weight_max_vals = compute_percentile(weight_expanded, weight_max)  # [1, channels, C]


    input_quant = input_quantizer.quantize(input_expanded, input_min_vals, input_max_vals) # [2048, 64, 60, C]
    weight_quant = weight_quantizer.quantize(weight_expanded, weight_min_vals, weight_max_vals)  # [180, 60, C]

    input_quant = input_quant.view(-1, input_tensor.size(2), num_combinations) #[2048, 64, 60, C] -> [2048 * 64, 60, C]
    input_quant = input_quant.permute(2, 0, 1) #[C, 2048 * 64, 60]
    weight_quant = weight_quant.permute(2, 1, 0)[None, ...] #[1, C, 60, 180]
    origin_batch = origin_output.reshape(1, 1, -1, origin_output.shape[-1]) #[1, 1, 2048*64, 180]

    best_score = 1e+10
    best_idx1, best_idx2 = 0, 0
    batch = len(percentiles)

    for i in range(0, num_combinations, batch):
        #input_batch = input_quant[i][None, ...]
        end_idx = min(i + batch, num_combinations)
        input_batch = torch.stack([input_quant[idx][None, ...] for idx in range(i, end_idx)], dim=0) #[batch, 1, ...]
        with torch.no_grad():
            output_batch = input_batch @ weight_quant + bias
        del input_batch
        loss = F.mse_loss(output_batch, origin_batch, reduction='none')
        loss = loss.mean(dim=(2, 3))

        min_mse_idx = torch.argmin(loss)
        idx1 = min_mse_idx // num_combinations
        idx2 = min_mse_idx % num_combinations
        min_loss = loss[idx1, idx2]

        if min_loss < best_score:
            best_score = min_loss
            best_idx1 = idx1 + i
            best_idx2 = idx2
        del output_batch, loss
        torch.cuda.empty_cache()
        input_quantizer.init_quantization_scale(input_min_vals[..., best_idx1],
                                             input_max_vals[..., best_idx1])
        weight_quantizer.init_quantization_scale(weight_min_vals[..., best_idx2],
                                             weight_max_vals[..., best_idx2])
    '''
        index = torch.arange(num_combinations).unsqueeze(-1).repeat(1, num_combinations).reshape(-1)
        input_quant = input_quant[..., index]
        input_min_vals = input_min_vals[..., index]
        input_max_vals = weight_max_vals[..., index]
        weight_quant = weight_quant.unsqueeze(-2).expand(-1, -1, num_combinations, num_combinations).reshape(*weight_quant.shape[:-1], -1)
        weight_min_vals = weight_min_vals.unsqueeze(-2).expand(-1, -1, num_combinations, num_combinations).reshape(*weight_min_vals.shape[:-1], -1)
        weight_max_vals = weight_max_vals.unsqueeze(-2).expand(-1, -1, num_combinations, num_combinations).reshape(*weight_max_vals.shape[:-1], -1)

        num_combinations *= num_combinations
        input_quant = input_quant.view(-1, input_tensor.size(2), num_combinations) #[2048, 64, 60, C] -> [2048 * 64, 60, C]
        weight_quant = weight_quant.permute(2, 1, 0).cuda() # [C, 60, 180]
        input_quant = input_quant.permute(2, 0, 1).cuda() #[C, 2048 * 64, 60]

        batch_size = 64
        best_score = 1e+10
        best_idx = 0

        for i in range(0, num_combinations, batch_size):
            end_idx = min(i + batch_size, num_combinations)
            input_batch = input_quant[i:end_idx]
            weight_batch = weight_quant[i:end_idx]

            with torch.no_grad():
                output_batch = torch.bmm(input_batch, weight_batch)
            output_batch = output_batch.view((end_idx - i), b, n, out_c) + bias
            origin_batch = origin_output.clone().detach().unsqueeze(0).expand((end_idx - i), -1, -1, -1)

            loss = F.mse_loss(output_batch, origin_batch, reduction='none')
            loss = loss.mean(dim=(1, 2, 3))

            min_loss, idx = loss.min(dim=0)
            if min_loss < best_score:
                best_score = min_loss
                best_idx = idx

            del input_batch, weight_batch, output_batch, loss
            torch.cuda.empty_cache()

        weight_quantizer.init_quantization_scale(weight_min_vals.reshape(num_combinations, -1)[best_idx],
                                                weight_max_vals.reshape(num_combinations, -1)[best_idx])
        input_quantizer.init_quantization_scale(input_min_vals.reshape(num_combinations, -1)[best_idx],
                                                input_max_vals.reshape(num_combinations, -1)[best_idx])
                                             '''

    return best_score

def search_log2_linear_quantizer(input_quantizer, weight_quantizer, origin_output, x, weight, bias):
    input_tensor = x.clone().detach()
    weight_tensor = weight.clone().detach()

    percentiles = torch.tensor([0.99, 0.999, 0.9999, 0.99999])

    combinations = list(itertools.product(percentiles, percentiles)) 
    combinations = torch.tensor(combinations)

    input_delta = combinations[:, 0]
    weight_delta = combinations[:, 1]

    num_combinations = input_delta.size(0)
    b, n, _ = input_tensor.shape
    out_c = weight_tensor.shape[0]

    input_expanded = input_tensor.unsqueeze(-1).expand(-1, -1, -1, num_combinations)  # [2048, 64, 60, C]
    weight_expanded = weight_tensor.unsqueeze(-1).expand(-1, -1, num_combinations)  # [180, 60, C]

    input_max_vals = compute_percentile(input_expanded, input_delta, log=True) # [1, 1, channels, C]
    weight_max_vals = compute_percentile(weight_expanded, weight_delta, log=True) # [1, channels, C]

    input_quant = input_quantizer.quantize(input_expanded, input_max_vals) # [2048, 64, 60, C]
    weight_quant = weight_quantizer.quantize(weight_expanded, weight_max_vals)  # [180, 60, C]
    
    input_quant = input_quant.view(-1, input_tensor.size(2), num_combinations) #[2048, 64, 60, C] -> [2048 * 64, 60, C]
    input_quant = input_quant.permute(2, 0, 1) #[C, 2048 * 64, 60]
    weight_quant = weight_quant.permute(2, 1, 0)[None, ...] #[1, C, 60, 180]
    origin_batch = origin_output.reshape(1, 1, -1, origin_output.shape[-1]) #[1, 1, 2048*64, 180]

    best_score = 1e+10
    best_idx1, best_idx2 = 0, 0
    batch = len(percentiles)

    for i in range(0, num_combinations, batch):
        #input_batch = input_quant[i][None, ...]
        end_idx = min(i + batch, num_combinations)
        input_batch = torch.stack([input_quant[idx][None, ...] for idx in range(i, end_idx)], dim=0) #[batch, 1, ...]

        with torch.no_grad():
            output_batch = input_batch @ weight_quant + bias
        del input_batch
        loss = F.mse_loss(output_batch, origin_batch, reduction='none')
        loss = loss.mean(dim=(2, 3))

        min_mse_idx = torch.argmin(loss)
        idx1 = min_mse_idx // num_combinations
        idx2 = min_mse_idx % num_combinations
        min_loss = loss[idx1, idx2]

        if min_loss < best_score:
            best_score = min_loss
            best_idx1 = idx1 + i
            best_idx2 = idx2
        del output_batch, loss
        torch.cuda.empty_cache()

        input_quantizer.init_quantization_scale(input_max_vals[..., best_idx1])
        weight_quantizer.init_quantization_scale(weight_max_vals[..., best_idx2])

    '''
        input_quant = input_quant.view(-1, input_tensor.size(2), num_combinations) #[2048, 64, 60, C] -> [2048 * 64, 60, C]
        weight_quant = weight_quant.permute(2, 1, 0).cuda() # [C, 60, 180]
        input_quant = input_quant.permute(2, 0, 1).cuda() #[C, 2048 * 64, 60]

        batch_size = 64
        best_score = 1e+10
        best_idx = 0

        for i in range(0, num_combinations, batch_size):
            end_idx = min(i + batch_size, num_combinations)
            input_batch = input_quant[i:end_idx]
            weight_batch = weight_quant[i:end_idx]

            with torch.no_grad():
                output_batch = torch.bmm(input_batch, weight_batch)
            output_batch = output_batch.view((end_idx - i), b, n, out_c) + bias
            origin_batch = origin_output.clone().detach().unsqueeze(0).expand((end_idx - i), -1, -1, -1)

            loss = F.mse_loss(output_batch, origin_batch, reduction='none')
            loss = loss.mean(dim=(1, 2, 3))
            min_loss, idx = loss.min(dim=0)

            if min_loss < best_score:
                best_score = min_loss
                best_idx = idx

            del input_batch, weight_batch, output_batch, loss
            torch.cuda.empty_cache()

        input_quantizer.init_quantization_scale(input_max_vals.reshape(num_combinations, -1)[best_idx])
        weight_quantizer.init_quantization_scale(weight_max_vals.reshape(num_combinations, -1)[best_idx])
    '''

    return best_score

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
        best_name = ""
        for name, module in self.quantizers_dict.items():
            if name == "origin_module" or name == "best_module":
                continue
            loss = module.search_best_setting(origin_output, *args)
            if loss < best_score:
                best_score = loss
                self.quantizers_dict["best_module"] = module
                best_name = name
            print("finish one quantizer")
        print(best_name)
        print("finish one module")

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

            linear_quantizers = List_Quantizers().cuda()

            linear_quantizers.append_quantizer("origin_module", m)
            linear_quantizers.append_quantizer("best_module", m)
            for quantizer_info in quantizers:
                input_quantizer = quantizer_info["quantizer1"]
                weight_quantizer = quantizer_info["quantizer2"]

                new_m = QuantLinear(m.in_features, m.out_features, input_quantizer, weight_quantizer)
                new_m.weight.data = m.weight.data
                new_m.bias = m.bias

                linear_quantizers.append_quantizer(quantizer_info["name"], new_m)
            linear_quantizers = linear_quantizers
            setattr(father_module, name[idx:], linear_quantizers)
        elif isinstance(m, MatMul):
            # Matmul Layer
            idx = idx + 1 if idx != 0 else idx

            quantizers = create_quantizers(quant_params)

            matmul_quantizers = List_Quantizers().cuda()

            matmul_quantizers.append_quantizer("origin_module", m)
            matmul_quantizers.append_quantizer("best_module", m)
            for quantizer_info in quantizers:
                quantizer1 = quantizer_info["quantizer1"]
                quantizer2 = quantizer_info["quantizer2"]

                new_m = QuantMatMul(quantizer1, quantizer2)

                matmul_quantizers.append_quantizer(quantizer_info["name"], new_m)
            #matmul_quantizers = matmul_quantizers.to(m.device)
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
        self.print_network(self.net_g)

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
