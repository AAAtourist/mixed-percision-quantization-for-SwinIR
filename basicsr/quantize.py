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

def change_singular(weight, x):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lambda_1 = 100
    rho = 50
    max_iter = 100
    weight = weight.T

    W = weight.clone().detach()
    Z = W.clone().detach()
    U = torch.zeros_like(W)

    X_reshape = x.reshape(-1, x.shape[-1])
    Y_reshape = X_reshape @ weight.to(device)

    XTX = X_reshape.t() @ X_reshape
    I_n = torch.eye(X_reshape.shape[1], device=device)
    A = XTX + rho*I_n
    A_inv = torch.inverse(A)
    XTY = X_reshape.t() @ Y_reshape

    r = torch.tensor(60)
    r_10 = int(torch.ceil(0.1*r).item())
    r_20 = int(torch.ceil(0.2*r).item())
    r_80 = int(torch.ceil(0.8*r).item())
    r_90 = int(torch.ceil(0.9*r).item())
    w = torch.ones(r)
    # 前10%
    w[:r_10] = 100
    # 前10%-20%
    w[r_10:r_20] = 10
    # 中间60%已经是1了
    # 后20%-10%
    w[r_80:r_90] = 1
    # 最后10%
    w[r_90:] = 1
    #w = torch.ones(r)

    w = w.to(device)
    for it in range(max_iter + 1):
        # W-update
        RHS = XTY + rho*(Z - U)
        W = A_inv @ RHS

        # Z-update
        W_plus_U = W + U
        #W_plus_U = W
        P, sigma_W, Qt = torch.linalg.svd(W_plus_U, full_matrices=False)
        r = len(sigma_W)


        # Compute target singular value t
        #t = torch.mean(sigma_W)
        t = 2


        # Update singular values
        sigma_Z = (rho * sigma_W + lambda_1 * t * w) / (rho + lambda_1*w)
        sigma_Z = torch.maximum(sigma_Z, torch.tensor(0))

        # Reconstruct Z
        Z = P @ torch.diag(sigma_Z) @ Qt


        # U-update (scaled dual variable)
        U = U + (W - Z)

        # 检查收敛
        if it % 10 == 0:
            res = F.mse_loss(X_reshape @ W, Y_reshape)
            primal_res = torch.norm(W - Z, p='fro')
            print(f"res: {res.item():.6f}")
            print(f"Iter {it}, primal_res: {primal_res.item():.6f}")
    
    return W.T

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

    def per_channel_bound(self, shape):
        if len(shape) == 2:
            self.lower_bound = nn.Parameter(self.lower_bound.expand(shape[0], 1))
            self.upper_bound = nn.Parameter(self.upper_bound.expand(shape[0], 1))
        elif len(shape == 3):
            self.lower_bound = nn.Parameter(self.lower_bound.expand(1, 1, shape[-1]))
            self.upper_bound = nn.Parameter(self.upper_bound.expand(1, 1, shape[-1]))
        else:
            self.lower_bound = nn.Parameter(self.lower_bound.expand(1, shape[1], 1, shape[-1]))
            self.upper_bound = nn.Parameter(self.upper_bound.expand(1, shape[1], 1, shape[-1]))
        
class UniformQuantizer(QuantizerBase):
    def __init__(self, n_bits: int = 8):
        super(UniformQuantizer, self).__init__(n_bits=n_bits)

    def forward(self, x: torch.Tensor):
        
        delta = (self.upper_bound - self.lower_bound) / (self.n_levels - 1)

        x_cut = self.clip(x, self.lower_bound, self.upper_bound)

        x_int = self.round((x_cut - self.lower_bound) / delta)
    
        return x_int * delta + self.lower_bound
    
    
    def init_quantization_scale(self, x, one_direction_search = False):
        lb, ub = DOBI(x, bit=self.n_bits, one_direction=one_direction_search, is_uni=True)
        self.set_params_lb_manually(lb)
        self.set_params_ub_manually(ub)

class Log2Quantizer(QuantizerBase):
    def __init__(self, n_bits: int = 8):
        super(Log2Quantizer, self).__init__(n_bits=n_bits)

    def forward(self, x: torch.Tensor):


        delta = self.upper_bound - self.lower_bound

        x_cut = self.clip(x, self.lower_bound + 1e-6, self.upper_bound)

        x_int = self.round(-1 * torch.log2((x_cut - self.lower_bound) / delta))

        mask = x_int >= self.n_levels

        x_int[mask] = self.n_levels - 1
    
        x_float_q = 2**(-1 * x_int) * delta + self.lower_bound

        x_float_q[mask] = self.lower_bound

        return x_float_q
    def init_quantization_scale(self, x, one_direction_search = False):
        lb, ub = DOBI(x, bit=self.n_bits, one_direction=one_direction_search, is_uni=False)
        self.set_params_lb_manually(lb)
        self.set_params_ub_manually(ub)

class QuantLinear(nn.Linear):
    def __init__(self,
                 in_features,
                 out_features,
                 dic_input_quantizer=None,
                 dic_weight_quantizer=None):
        super(QuantLinear, self).__init__(in_features, out_features)

        
        self.dic_input_quantizer = dic_input_quantizer
        self.dic_weight_quantizer = dic_weight_quantizer
        self.bit = 4
        self.quant_weight = None

        self.first_time = True

    def forward(self, x):
        if self.first_time:
            self.search_best_setting(x)
            #self.quant_weight = self.dic_weight_quantizer[f"{self.bit}"](self.weight)
            if 120 not in self.weight.shape:
                self.weight = nn.Parameter(change_singular(self.weight, x))
            self.first_time = False

            return F.linear(x, weight = self.weight, bias = self.bias)

        quant_x = self.dic_input_quantizer[f"{self.bit}"](x)
        self.quant_weight = self.dic_weight_quantizer[f"{self.bit}"](self.weight)
        out = F.linear(quant_x, weight = self.quant_weight, bias = self.bias)

        return out

    def search_best_setting(self, x):#cali
        print('start linear cali')
        self.dic_input_quantizer[f"{self.bit}"].init_quantization_scale(x, one_direction_search = True)
        self.dic_weight_quantizer[f"{self.bit}"].init_quantization_scale(self.weight, one_direction_search = False)

        self.dic_weight_quantizer[f"{self.bit}"].per_channel_bound(x.shape)
        self.dic_weight_quantizer[f"{self.bit}"].per_channel_bound(self.weight.shape)

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
        self.dic_input_quantizer[f"{self.bit}"].init_quantization_scale(A, one_direction_search = True)
        self.dic_weight_quantizer[f"{self.bit}"].init_quantization_scale(B, one_direction_search = False)

        self.dic_weight_quantizer[f"{self.bit}"].per_channel_bound(A.shape)
        self.dic_weight_quantizer[f"{self.bit}"].per_channel_bound(B.shape)

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

        self.new_qk = QuantLinear(self.features, self.features * 2, dic_input_qk, dic_weight_qk).cuda()
        #self.new_q = QuantLinear(self.features, self.features, dic_input_q, dic_weight_q, True).cuda()
        #self.new_k = QuantLinear(self.features, self.features, dic_input_k, dic_weight_k, True).cuda()
        self.new_v = QuantLinear(self.features, self.features, dic_input_v, dic_weight_v).cuda()

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

            if 'qkv' in name:
                new_m = qkv_module(m, quant_params)
            else:
                dic_input_q, dic_weight_q = create_quantizers(False, quant_params)
                new_m = QuantLinear(m.in_features, m.out_features, dic_input_q, dic_weight_q).cuda()
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
