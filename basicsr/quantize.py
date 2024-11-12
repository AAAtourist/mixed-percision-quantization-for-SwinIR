import itertools
import torch
import numpy as np
import torch.optim
from torch.autograd.function import _ContextMethodMixin
from torch.autograd import Function
from torch import Tensor, FloatTensor
from basicsr.models import lr_scheduler as lr_scheduler
import torch.nn as nn
from basicsr.archs.swinir_arch import SwinTransformerBlock, WindowAttention, MatMul
from torch.nn import functional as F
from typing import Any
from basicsr.smooth_networks import smooth_network

num_linear = 0
num_matmul = 0

def lp_loss(pred, tgt, p=2.0):
    return (pred-tgt).abs().pow(p).mean()

class Differentiable_Round(Function):
    @staticmethod
    def forward(ctx: _ContextMethodMixin, x: Tensor):
        return x.round()

    @staticmethod
    def backward(ctx: _ContextMethodMixin, grad_outputs):
        return grad_outputs

class Differentiable_Clip(Function):
    @staticmethod
    def forward(
        ctx: _ContextMethodMixin,
        input: Tensor,
        min_val: Tensor,
        max_val: Tensor,
    ) -> Any:
        assert isinstance(max_val, Tensor), 'nononno'
        ctx.save_for_backward(input, min_val, max_val)
        # if isinstance(min_val, Tensor):
        #     min_val = min_val.item()
        # if isinstance(max_val, Tensor):
        #     max_val = max_val.item()
        
        # output = input.clamp(min_val, max_val)
        output = input.clamp(min_val.item(), max_val.item())
        return output

    @staticmethod
    def backward(ctx: _ContextMethodMixin, grad_outputs: Tensor) -> Any:
        input, min_val, max_val = ctx.saved_tensors

        grad_input = grad_outputs.clone()
        grad_input[(input < min_val) | (input > max_val)] = 0
        
        grad_min = grad_outputs.clone()
        grad_min[input > min_val] = 0
        grad_min = grad_min.sum().view(1)

        grad_max = grad_outputs.clone()
        grad_max[input < max_val] = 0
        grad_max = grad_max.sum().view(1)
        return grad_input, grad_min, grad_max

def quant(input:torch.Tensor, lb:float, ub:float, bit:int, is_uni:bool):
    if is_uni == True:
        input = input.clamp(lb, ub)
        s = (ub - lb) / (2 ** bit -1)
        input = (input - lb)/s
        input = input.round()
        input = input * s + lb
    else:
        input = input.clamp(lb + 1e-6, ub)
        s = ub - lb
        input = torch.round(-1 * torch.log2((input - lb) / s))
        mask = input >= 2 ** bit
        input = torch.clamp(input, 0, 2 ** bit - 1)
        input = 2**(-1 * input) * s + lb
        input[mask] = lb
    return input

def cal_mse(input:torch.Tensor, lb:float, ub:float, bit:int, is_uni:bool):
    quant_input = quant(input, lb, ub, bit, is_uni)
    res = float(torch.norm(input - quant_input))
    return res

def DOBI(input:torch.Tensor, bit:int, one_direction = False, is_uni = True, num:int=100):
    min_value = torch.min(input)
    max_value = torch.max(input)
    
    diff = (max_value - min_value) / (2 * num)
    
    history_min = float('inf')
    input = input.cuda()
    
    if one_direction:
        diff = (max_value - min_value) / num
        for i in range(num):
            lb = min_value
            ub = max_value - diff * i
            cur_value = cal_mse(input, lb, ub, bit, is_uni)
            if cur_value < history_min:
                best_lb = lb
                best_ub = ub
                history_min = cur_value
    else:
        diff = (max_value - min_value) / (2 * num)
        for i in range(num):
            lb = min_value + diff * i
            ub = max_value - diff * i
            cur_value = cal_mse(input, lb, ub, bit, is_uni)
            if cur_value < history_min:
                best_lb = lb
                best_ub = ub
                history_min = cur_value
    
    return float(best_lb), float(best_ub)

class QuantizerBase(nn.Module):
    def __init__(self, n_bits: int = 4):
        super(QuantizerBase, self).__init__()
        self.lower_bound = nn.Parameter(
            torch.randn((1,), dtype=torch.float32),
        )
        self.upper_bound = nn.Parameter(
            torch.randn((1,), dtype=torch.float32),
        )
        
        assert 2 <= n_bits <= 8, 'bitwidth not supported'
        self.n_bits = n_bits
        self.n_levels = 2 ** self.n_bits

        self.clip = Differentiable_Clip.apply
        self.round = Differentiable_Round.apply

        self.calibrated = False
        self.one_direction_search = False

    def set_params_lb_manually(self, lb: Tensor):
        device = self.lower_bound.device
        self.lower_bound.data = FloatTensor([lb]).data.clone().to(device)
    
    def set_params_ub_manually(self, ub: Tensor):
        device = self.upper_bound.device
        self.upper_bound.data = FloatTensor([ub]).data.clone().to(device)

class UniformQuantizer(QuantizerBase):
    def __init__(self, n_bits: int = 8):
        super(UniformQuantizer, self).__init__(n_bits=n_bits)

    def forward(self, x: torch.Tensor):
        
        # start quantization
        #if self.upper_bound.device != x.device:
        #    self.upper_bound = self.upper_bound.to(x.device)
        #    self.lower_bound = self.lower_bound.to(x.device)

        delta = (self.upper_bound - self.lower_bound) / (self.n_levels - 1)

        x_cut = self.clip(x, self.lower_bound, self.upper_bound)

        x_int = self.round((x_cut - self.lower_bound) / delta)
    
        return x_int * delta + self.lower_bound
    
    '''
        def quantize(self, x, max, min):
            x_clone = x.clone().detach()
            delta = (max - min) / (2 ** self.n_bits - 1)
            zero_point = (- min / delta).round()
            x_int = torch.round(x_clone / delta)
            x_quant = torch.clamp(x_int + zero_point, 0, self.n_levels - 1)
            x_float_q = (x_quant - zero_point) * delta
            return x_float_q
            '''
    
    def init_quantization_scale(self, x, one_direction_search = False):
        lb, ub = DOBI(x, bit=self.n_bits, one_direction=one_direction_search, is_uni=True)
        self.set_params_lb_manually(lb)
        self.set_params_ub_manually(ub)

        '''
            delta, zero_point = None, None
            if channel_wise:
                x_clone = x.clone().detach()
                n_channels = x_clone.shape[-1] if len(x.shape) != 4 else x_clone.shape[-1] * x_clone.shape[1]
                delta = torch.zeros(n_channels)
                zero_point = torch.zeros(n_channels)

                for c in range(n_channels):
                    if len(x.shape) == 3:
                        delta[c], zero_point[c] = self.init_quantization_scale(x_clone[:,:,c], channel_wise=False)
                    elif len(x.shape) == 4:
                        delta[c], zero_point[c] = self.init_quantization_scale(x_clone[:,c // x.shape[-1],:, c % x.shape[-1]],
                                                                                channel_wise=False)
                    else:
                        delta[c], zero_point[c] = self.init_quantization_scale(x_clone[:,c], channel_wise=False)

                if len(x.shape) == 4:
                    delta = delta.view(1, x.shape[1], 1, -1)
                    zero_point = zero_point.view(1, x.shape[1], 1, -1)
                elif len(x.shape) == 2:#[180 60]
                    delta = delta.view(1, -1)
                    zero_point = zero_point.view(1, -1)
                elif len(x.shape) == 3:
                    delta = delta.view(1, 1, -1)
                    zero_point = zero_point.view(1, 1, -1)
                else:
                    raise NotImplementedError
                
                self.delta = delta
                self.zero_point = zero_point
            else:
                x_clone = x.clone().detach()
                x_max = x_clone.max()
                x_min = x_clone.min()
                best_score = 1e+10
                for pct1 in [0.85, 0.9, 0.99, 0.999, 0.9999]:
                    for pct2 in [0.15, 0.1, 0.01, 0.001, 0.0001]:
                        try:
                            new_max = torch.quantile(x_clone.reshape(-1), pct1)
                            new_min = torch.quantile(x_clone.reshape(-1), pct2)
                        except:
                            new_max = torch.tensor(np.percentile(
                                x_clone.reshape(-1).cpu(), pct1 * 100),
                                device=x_clone.device,
                                dtype=torch.float32)
                            new_min = torch.tensor(np.percentile(
                                x_clone.reshape(-1).cpu(), pct2 * 100),
                                device=x_clone.device,
                                dtype=torch.float32)   
                        x_q = self.quantize(x_clone, new_max, new_min)
                        score = lp_loss(x_clone, x_q, p=2)
                        if score < best_score:
                            best_score = score
                            delta = (new_max - new_min) / (2 ** self.n_bits - 1)
                            zero_point = (- new_min / delta).round()

                self.delta = delta
                self.zero_point = zero_point

            return delta, zero_point 
        '''

class Log2Quantizer(QuantizerBase):
    def __init__(self, n_bits: int = 8):
        super(Log2Quantizer, self).__init__(n_bits=n_bits)

    def forward(self, x: torch.Tensor):

        # start quantization
        #if self.upper_bound.device != x.device:
        #    self.upper_bound = self.upper_bound.to(x.device)
        #    self.lower_bound = self.lower_bound.to(x.device)

        delta = self.upper_bound - self.lower_bound

        x_cut = self.clip(x, self.lower_bound + 1e-6, self.upper_bound)

        x_int = self.round(-1 * torch.log2((x_cut - self.lower_bound) / delta))

        mask = x_int >= self.n_levels

        x_int[mask] = self.n_levels - 1
    
        x_float_q = 2**(-1 * x_int) * delta + self.lower_bound

        x_float_q[mask] = self.lower_bound

        return x_float_q

    '''
        def quantize(self, x, delta):
            x = x.to(delta.device)
        
            x_clone = x.clone().detach()

            x_min = x_clone.min()
            
            x_shifted = x_clone - x_min

            x_int = torch.round(-1 * torch.log2(torch.clamp(x_shifted / delta, min=1e-6, max=1)))

            mask = x_int >= self.n_levels
            
            x_quant = torch.clamp(x_int, min=0, max=self.n_levels - 1)

            x_float_q = 2**(-1 * x_quant) * delta + x_min

            x_float_q[mask] = 0
            
            return x_float_q
            '''

    def init_quantization_scale(self, x, one_direction_search = False):
        lb, ub = DOBI(x, bit=self.n_bits, one_direction=one_direction_search, is_uni=False)
        self.set_params_lb_manually(lb)
        self.set_params_ub_manually(ub)
        '''
            delta = None
            x_clone = x.clone().detach()
            delta = x_clone.max()
            best_score = 1e+10
            for pct in [0.9, 0.99, 0.999, 0.9999, 0.99999]:
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

            self.delta = delta
        '''

class QuantLinear(nn.Linear):
    def __init__(self,
                 in_features,
                 out_features,
                 dic_input_quantizer=None,
                 dic_weight_quantizer=None,
                 need_smooth = False):
        super(QuantLinear, self).__init__(in_features, out_features)

        
        self.dic_input_quantizer = dic_input_quantizer
        self.dic_weight_quantizer = dic_weight_quantizer
        self.bit = 4
        self.quant_weight = None
        self.need_smooth = need_smooth

        self.smooth_network = nn.Module()

        self.first_time = True

    def forward(self, x):
        if self.first_time:
            self.search_best_setting(x)
            self.quant_weight = self.dic_weight_quantizer[f"{self.bit}"](self.weight)
            self.first_time = False

        if self.need_smooth:
            origin_shape = x.shape[:-1]
            x = x.reshape(-1, 64, x.shape[-1])
            assert not torch.isnan(x).any(), 'nan x'
            #XA, BW, _ = self.smooth_network(x)
            #quant_XA = self.dic_input_quantizer[f"{self.bit}"](XA)
            #quant_BW = self.dic_weight_quantizer[f"{self.bit}"](BW)
            #quant_XA = XA
            out, _ = self.smooth_network(x, self.dic_input_quantizer[f"{self.bit}"], self.dic_weight_quantizer[f"{self.bit}"])
            
            #out = torch.bmm(quant_XA, quant_BW)
            out = out.reshape(((*origin_shape, -1)))
            out += self.bias if self.bias is not None else 0
        else:
            quant_x = self.dic_input_quantizer[f"{self.bit}"](x)
            out = F.linear(quant_x, weight = self.quant_weight, bias = self.bias)

        return out

    def search_best_setting(self, x):#cali
        print('start linear cali')
        if self.need_smooth:
            global num_sm
            if 'num_sm' not in globals():
                num_sm = 0
            print(f'start cali smooth network {num_sm}')
            

            self.smooth_network = smooth_network(self.weight.T, 20, self.bit)
            self.smooth_network.inited(x)
            #self.add_module("smooth_network", self.smooth_network)
            self.smooth_network.eval()

            #XA, BW, _ = self.smooth_network(x)
            #self.dic_input_quantizer[f"{self.bit}"].init_quantization_scale(XA) #per-tensor
            #self.dic_weight_quantizer[f"{self.bit}"].init_quantization_scale(BW)
            
            print(f'finish cali smooth network {num_sm}')
            num_sm += 1

        self.dic_input_quantizer[f"{self.bit}"].init_quantization_scale(x, one_direction_search = True)
        self.dic_weight_quantizer[f"{self.bit}"].init_quantization_scale(self.weight, one_direction_search = False)

    def get_bound_param(self):
        params = [self.dic_input_quantizer[f"{self.bit}"].lower_bound, self.dic_input_quantizer[f"{self.bit}"].upper_bound,
                self.dic_weight_quantizer[f"{self.bit}"].lower_bound, self.dic_weight_quantizer[f"{self.bit}"].upper_bound]

        for param in params:
            assert param.requires_grad, "One of the model's bound paramsters does not require gradients!"
        
        return params

class QuantMatMul(nn.Module):
    def __init__(self, 
                 dic_input_quantizer=None,
                 dic_weight_quantizer=None):
        super(QuantMatMul, self).__init__()

        self.dic_input_quantizer = dic_input_quantizer
        self.dic_weight_quantizer = dic_weight_quantizer
        self.bit = 4

        self.first_time = True
    
    def forward(self, A, B):
        if self.first_time:
            self.search_best_setting(A, B)
            self.first_time = False

        A = self.dic_input_quantizer[f"{self.bit}"](A)
        B = self.dic_weight_quantizer[f"{self.bit}"](B)
        out = A @ B
        return out

    def search_best_setting(self, A, B):
        print('start matmul search')
        
        #self.dic_input_quantizer[f"{self.bit}"].init_quantization_scale(A, True)
        #self.dic_weight_quantizer[f"{self.bit}"].init_quantization_scale(B, True)

        self.dic_input_quantizer[f"{self.bit}"].init_quantization_scale(A, one_direction_search = True)
        self.dic_weight_quantizer[f"{self.bit}"].init_quantization_scale(B, one_direction_search = False)

        print('finish matmul search')
        

    def get_bound_param(self):
        params = [self.dic_input_quantizer[f"{self.bit}"].lower_bound, self.dic_input_quantizer[f"{self.bit}"].upper_bound,
                self.dic_weight_quantizer[f"{self.bit}"].lower_bound, self.dic_weight_quantizer[f"{self.bit}"].upper_bound]

        for param in params:
            assert param.requires_grad, "One of the model's bound paramsters does not require gradients!"
        
        return params

def create_quantizers(need_log_quantizer=False, default_quant_params={}):
    #dic_input_q, dic_weight_q = {}, {}
    dic_input_q = nn.ModuleDict()
    dic_weight_q = nn.ModuleDict()

    for i in default_quant_params['bits_candidate']:
        input_param = {**default_quant_params["input_quant_params"], "n_bits": i}
        weight_param = {**default_quant_params["weight_quant_params"], "n_bits": i}

        if need_log_quantizer:
            #dic_input_q.append({f"{i}": Log2Quantizer(**input_param)})
            dic_input_q[f"{i}"] = Log2Quantizer(**input_param)
            #dic_input_q.append({"uni{i}": UniformQuantizer(**input_param)})
        else:
            dic_input_q[f"{i}"] = UniformQuantizer(**input_param)
        
        dic_weight_q[f"{i}"] = UniformQuantizer(**weight_param)
    
    return dic_input_q, dic_weight_q

'''class List_Quantizers(nn.Module):
    def __init__(self, name):
        super(List_Quantizers, self).__init__()

        self.quant_module_dict = nn.ModuleDict()
        self.need_search = True
        self.name = name

    def forward(self, *args):

        if self.need_search:
            origin_output = self.quant_module_dict["origin_module"](*args)
            self.module_search(origin_output, *args)
            self.need_search = False
            return origin_output

        return self.quant_module_dict["quant_module"](*args)

    def append_quantizer(self, name, module):
        self.quant_module_dict[name] = module

    def module_search(self, origin_output, *args):
        global num_module
        if 'num_module' not in globals():
            num_module = 0

        best_score = 1e+10
        for name, module in self.quant_module_dict.items():
            if name == "origin_module" :
                continue
            module.search_best_setting(origin_output, *args)

        print(f"finish one module{num_module}")
        num_module += 1
'''
class qkv_module(nn.Module):
    def __init__(self, origin_linear=None, quant_params={}):
        super(qkv_module, self).__init__()

        self.features = origin_linear.in_features  #60
        origin_linear.out_features  #180
        weight = origin_linear.weight.data #180 60
        bias = origin_linear.bias #180
        dic_input_qk, dic_weight_qk = create_quantizers(False, quant_params)
        #dic_input_q, dic_weight_q = create_quantizers(False, quant_params)
        #dic_input_k, dic_weight_k = create_quantizers(False, quant_params)
        dic_input_v, dic_weight_v = create_quantizers(False, quant_params)

        self.new_qk = QuantLinear(self.features, self.features * 2, dic_input_qk, dic_weight_qk, True).cuda()
        #self.new_q = QuantLinear(self.features, self.features, dic_input_q, dic_weight_q, True).cuda()
        #self.new_k = QuantLinear(self.features, self.features, dic_input_k, dic_weight_k, True).cuda()
        self.new_v = QuantLinear(self.features, self.features, dic_input_v, dic_weight_v, True).cuda()

        self.new_qk.bias = nn.Parameter(bias[:self.features * 2])
        #self.new_q.bias = nn.Parameter(bias[:self.features])
        #self.new_k.bias = nn.Parameter(bias[self.features : 2 * self.features])
        self.new_v.bias = nn.Parameter(bias[2 * self.features : 3 * self.features])

        self.new_qk.weight.data = nn.Parameter(weight[:self.features * 2, :])
        #self.new_q.weight.data = nn.Parameter(weight[: self.features, :])
        #self.new_k.weight.data = nn.Parameter(weight[self.features : 2 * self.features, :])
        self.new_v.weight.data = nn.Parameter(weight[2 * self.features : 3 * self.features, :])

    def forward(self, x):
        
        assert not torch.isnan(x).any(), 'nan qkv'

        outqk = self.new_qk(x)
        #outq = self.new_q(x)
        #outk = self.new_k(x)
        outv = self.new_v(x)

        return torch.cat((outqk, outv), dim=-1)

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

            need_smooth = True if 'qkv' in name or 'proj' in name else False
            #need_smooth = True
            if 'qkv' in name:
                new_m = qkv_module(m, quant_params)
            else:
                dic_input_q, dic_weight_q = create_quantizers(False, quant_params)
                new_m = QuantLinear(m.in_features, m.out_features, dic_input_q, dic_weight_q, need_smooth).cuda()
                new_m.weight.data = m.weight.data
                new_m.bias = m.bias
            
            setattr(father_module, name[idx:], new_m)

        elif isinstance(m, MatMul):
            idx = idx + 1 if idx != 0 else idx

            need_log_quantizer = True if 'MatMul2' in name else False

            dic_input_q, dic_weight_q = create_quantizers(need_log_quantizer, quant_params)

            new_m = QuantMatMul(dic_input_q, dic_weight_q).cuda()

            setattr(father_module, name[idx:], new_m)

    return model
