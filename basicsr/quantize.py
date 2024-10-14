import itertools
import torch


import numpy as np
import torch
import torch.optim

from basicsr.models import lr_scheduler as lr_scheduler
import torch.nn as nn

from basicsr.archs.swinir_arch import SwinTransformerBlock, WindowAttention, MatMul
from torch.nn import functional as F

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

        x_min = torch.min(x.detach()) - epsilon
        x -= x_min

        x_int = torch.round(-1 * (x / delta).log2())

        mask = x_int >= self.n_levels
        x_quant = torch.clamp(x_int, 0, self.n_levels - 1)

        x_float_q = 2**(-1 * x_quant) * delta + x_min

        x_float_q[mask] = 0
        
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
                 dic_input_quantizer=None,
                 dic_weight_quantizer=None):
        super(QuantLinear, self).__init__(in_features, out_features)

        
        self.dic_input_quantizer = dic_input_quantizer
        self.dic_weight_quantizer = dic_weight_quantizer
        self.bit = None
        self.quant_weight = None

        self.first_time = True

    def forward(self, x):
        if self.first_time:
            global num_linear
            print(f'one linear finish:{num_linear}')
            num_linear += 1
            self.first_time = False
            self.quant_weight = self.dic_weight_quantizer["{self.bit}"](self.weight)
        
        x = self.dic_input_quantizer["{self.bit}"](x)
        
        out = F.linear(x, weight=self.quant_weight, bias=self.bias)

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
                 dic_input_quantizer=None,
                 dic_weight_quantizer=None):
        super(QuantMatMul, self).__init__()

        self.dic_input_quantizer = dic_input_quantizer
        self.dic_weight_quantizer = dic_weight_quantizer
        self.bit = None

        self.first_time = True
    
    def forward(self, A, B):
        if self.first_time:
            global num_matmul
            print(f"one matmul finish:{num_matmul}")
            num_matmul += 1
            self.first_time = False

        A = self.dic_input_quantizer["{self.bit}"](A)
        B = self.dic_input_quantizer["{self.bit}"](B)
        
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

    return best_score

def create_quantizers(need_log_quantizer, default_quant_params={}):
    dic_input_q, dic_weight_q = [], []
    for i in default_quant_params['bits_candidate']:
        input_param = {**default_quant_params["input_quant_params"], "n_bits": i}
        weight_param = {**default_quant_params["weight_quant_params"], "n_bits": i}

        if need_log_quantizer:
            dic_input_q.append({"log{i}": Log2Quantizer(**input_param)})
            dic_input_q.append({"uni{i}": UniformQuantizer(**input_param)})
        else:
            dic_input_q.append({"{i}": UniformQuantizer(**input_param)})
        
        dic_weight_q.append({"{i}": UniformQuantizer(**weight_param)})
    
    return dic_input_q, dic_weight_q

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
            return origin_output

        return self.quantizers_dict["quant_module"](*args)

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

            need_log_quantizer = True if 'fc1' in name else 0

            dic_input_q, dic_weight_q = create_quantizers(need_log_quantizer, quant_params)

            new_m = QuantLinear(m.in_features, m.out_features, dic_input_q, dic_weight_q)
            new_m.weight.data = m.weight.data
            new_m.bias = m.bias

            linear_quantizers = List_Quantizers().cuda()

            linear_quantizers.append_quantizer("origin_module", m)
            linear_quantizers.append_quantizer("quant_module", new_m)
            
            setattr(father_module, name[idx:], linear_quantizers)
        elif isinstance(m, MatMul):
            # Matmul Layer
            idx = idx + 1 if idx != 0 else idx

            need_log_quantizer = True if 'MatMul2' in name else 0

            dic_input_q, dic_weight_q = create_quantizers(need_log_quantizer, quant_params)

            new_m = QuantMatMul(dic_input_q, dic_weight_q)
            
            matmul_quantizers = List_Quantizers().cuda()

            matmul_quantizers.append_quantizer("origin_module", m)
            matmul_quantizers.append_quantizer("quant_module", new_m)

            setattr(father_module, name[idx:], matmul_quantizers)

    return model
