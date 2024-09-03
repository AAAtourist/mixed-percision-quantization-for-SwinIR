from functools import partial
import os
from re import L
import time
from imageio import imopen
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
from torch.nn import Module
from torch import Tensor
from basicsr.utils.registry import MODEL_REGISTRY
from torch.optim import Adam

from basicsr.archs.swinir_arch import SwinTransformerBlock, WindowAttention
from basicsr.utils import get_root_logger, imwrite, tensor2img
from os import path as osp
from basicsr.metrics import calculate_metric

import functools
from torch.optim import Adam, SGD
from torchvision.transforms import Resize
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from types import MethodType
import torch
from tqdm import tqdm

import torch
import time
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.nn import Parameter
from copy import deepcopy

import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

num_linear = 0
num_matmul = 0

class MatMul(nn.Module):
    def forward(self, A, B):
        return A @ B

def lp_loss(pred, tgt, p=2.0, reduction='none'):
    """
    loss function measured in L_p Norm
    """
    if reduction == 'none':
        return (pred-tgt).abs().pow(p).sum(1).mean()
    else:
        return (pred-tgt).abs().pow(p).mean()


class UniformQuantizer(nn.Module):
    """
    PyTorch Function that can be used for asymmetric quantization (also called uniform affine
    quantization). Quantizes its argument in the forward pass, passes the gradient 'straight
    through' on the backward pass, ignoring the quantization that occurred.
    Based on https://arxiv.org/abs/1806.08342.
    :param n_bits: number of bit for quantization
    :param channel_wise: if True, compute scale and zero_point in each channel
    """
    def __init__(self, n_bits: int = 8, channel_wise: bool = False):
        super(UniformQuantizer, self).__init__()
        assert 2 <= n_bits <= 8, 'bitwidth not supported'
        self.n_bits = n_bits
        self.n_levels = 2 ** self.n_bits
        self.delta = None
        self.zero_point = None
        self.inited = False
        self.channel_wise = channel_wise
    
    def __repr__(self):
        s = super(UniformQuantizer, self).__repr__()
        s = "(" + s + " inited={}, channel_wise={})".format(self.inited, self.channel_wise)
        return s

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
                x_max = x_clone.abs().max(dim=-1)[0].max(dim=-1)[0].max(dim=-1)[0]
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
                else:
                    delta[c], zero_point[c] = self.init_quantization_scale(x_clone[c], channel_wise=False)
            if len(x.shape) == 4:
                delta = delta.view(-1, 1, 1, 1)
                zero_point = zero_point.view(-1, 1, 1, 1)
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
            for pct in [0.999, 0.9999, 0.99999]:
            # for pct in [0.9999, 0.99999]:
            # for pct in [0.999]:
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
                score = lp_loss(x_clone, x_q, p=2, reduction='all')
                if score < best_score:
                    best_score = score
                    delta = (new_max - new_min) / (2 ** self.n_bits - 1)
                    zero_point = (- new_min / delta).round()

        return delta, zero_point

    def quantize(self, x, max, min):
        delta = (max - min) / (2 ** self.n_bits - 1)
        zero_point = (- min / delta).round()
        # we assume weight quantization is always signed
        x_int = torch.round(x / delta)
        x_quant = torch.clamp(x_int + zero_point, 0, self.n_levels - 1)
        x_float_q = (x_quant - zero_point) * delta
        return x_float_q


class LogSqrt2Quantizer(nn.Module):
    """
    PyTorch Function that can be used for asymmetric quantization (also called uniform affine
    quantization). Quantizes its argument in the forward pass, passes the gradient 'straight
    through' on the backward pass, ignoring the quantization that occurred.
    Based on https://arxiv.org/abs/1806.08342.
    :param n_bits: number of bit for quantization
    :param channel_wise: if True, compute scale and zero_point in each channel
    """
    def __init__(self, n_bits: int = 8, channel_wise: bool = False):
        super(LogSqrt2Quantizer, self).__init__()
        assert 2 <= n_bits <= 8, 'bitwidth not supported'
        self.n_bits = n_bits
        self.n_levels = 2 ** self.n_bits
        self.delta = None
        self.inited = False
        self.channel_wise = channel_wise

    def forward(self, x: torch.Tensor):

        if self.inited is False:
            self.delta = self.init_quantization_scale(x)
            self.inited = True

        # start quantization
        x_dequant = self.quantize(x, self.delta)
        return x_dequant

    def init_quantization_scale(self, x: torch.Tensor):
        delta = None
        x_clone = x.clone().detach()
        delta = x_clone.max()
        best_score = 1e+10
        # for pct in [0.9999, 0.99999]: #
        for pct in [0.999, 0.9999, 0.99999]: #
        # for pct in [0.999]: #
            try:
                new_delta = torch.quantile(x_clone.reshape(-1), pct)
            except:
                new_delta = torch.tensor(np.percentile(
                    x_clone.reshape(-1).cpu(), pct * 100),
                    device=x_clone.device,
                    dtype=torch.float32)
            x_q = self.quantize(x_clone, new_delta)
            score = lp_loss(x_clone, x_q, p=2, reduction='all')
            if score < best_score:
                best_score = score
                delta = new_delta

        return delta

    def quantize(self, x, delta):      
        from math import sqrt
        x_int = torch.round(-1 * (x/delta).log2() * 2)
        mask = x_int >= self.n_levels
        x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
        odd_mask = (x_quant%2) * (sqrt(2)-1) + 1
        x_float_q = 2**(-1 * torch.ceil(x_quant/2)) * odd_mask * delta
        x_float_q[mask] = 0
        
        return x_float_q



class QuantLinear(nn.Linear):
    """
    Class to quantize weights of given Linear layer
    """
    def __init__(self,
                 in_features,
                 out_features,
                 input_quant_params={},
                 weight_quant_params={}):
        super(QuantLinear, self).__init__(in_features, out_features)

        self.input_quantizer = UniformQuantizer(**input_quant_params)
        self.weight_quantizer = UniformQuantizer(**weight_quant_params)

        self.use_input_quant = False
        self.use_weight_quant = False
        self.first_time = True

    def __repr__(self):
        s = super(QuantLinear, self).__repr__()
        s = "(" + s + "input_quant={}, weight_quant={})".format(self.use_input_quant, self.use_weight_quant)
        return s
    
    def set_quant_state(self, input_quant=False, weight_quant=False):
        self.use_input_quant = input_quant
        self.use_weight_quant = weight_quant

    def forward(self, x):
        """
        using quantized weights to forward input x
        """
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
    """
    Class to quantize weights of given Linear layer
    """
    def __init__(self,
                 input_quant_params={}):
        super(QuantMatMul, self).__init__()

        input_quant_params_matmul = deepcopy(input_quant_params)
        if 'log_quant' in input_quant_params_matmul:
            input_quant_params_matmul.pop('log_quant')
            self.quantizer_A = LogSqrt2Quantizer(**input_quant_params_matmul)
        else:
            self.quantizer_A = UniformQuantizer(**input_quant_params_matmul)
        self.quantizer_B = UniformQuantizer(**input_quant_params_matmul)

        self.use_input_quant = False
        self.first_time = True

    def __repr__(self):
        s = super(QuantMatMul, self).__repr__()
        s = "(" + s + "input_quant={})".format(self.use_input_quant)
        return s
    
    def set_quant_state(self, input_quant=False, weight_quant=False):
        self.use_input_quant = input_quant

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



def quant_model(model, input_quant_params={}, weight_quant_params={}):
    # post-softmax
    input_quant_params_matmul2 = deepcopy(input_quant_params)
    input_quant_params_matmul2['log_quant'] = True

    # SimQuant
    input_quant_params_channel = deepcopy(input_quant_params)
    input_quant_params_channel['channel_wise'] = True

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
            if 'qkv' in name or 'fc1' in name or 'reduction' in name:
                new_m = QuantLinear(m.in_features, m.out_features, input_quant_params_channel, weight_quant_params)
            else:
                new_m = QuantLinear(m.in_features, m.out_features, input_quant_params, weight_quant_params)
            new_m.weight.data = m.weight.data
            new_m.bias = m.bias
            setattr(father_module, name[idx:], new_m)
        elif isinstance(m, MatMul):
            # Matmul Layer
            idx = idx + 1 if idx != 0 else idx
            if 'matmul2' in name:
                new_m = QuantMatMul(input_quant_params_matmul2)
            else:
                new_m = QuantMatMul(input_quant_params)
            setattr(father_module, name[idx:], new_m)

    return model


def set_quant_state(model, input_quant=False, weight_quant=False):
    for m in model.modules():
        if isinstance(m, (QuantLinear, QuantMatMul)):
            m.set_quant_state(input_quant, weight_quant)

def window_attention_forward(self, x, mask = None):
    B_, N, C = x.shape
    qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

    q = q * self.scale
    # attn = (q @ k.transpose(-2, -1))
    attn = self.matmul1(q, k.transpose(-2,-1))

    relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
        self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
    relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    attn = attn + relative_position_bias.unsqueeze(0)

    if mask is not None:
        nW = mask.shape[0]
        attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
        attn = attn.view(-1, self.num_heads, N, N)
        attn = self.softmax(attn)
    else:
        attn = self.softmax(attn)

    attn = self.attn_drop(attn)

    # x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
    x = self.matmul2(attn, v).transpose(1, 2).reshape(B_, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x

@MODEL_REGISTRY.register()
class RepQModel:
    def __init__(self, opt):
        self.opt = opt
        self.logger = get_root_logger()
        self.device = torch.device("cuda:5" if opt["num_gpu"] != 0 else "cpu")
        self.is_train = opt["is_train"]
        self.phase = 1
        self.schedulers = []  # contains all the scheduler
        self.optimizers = []  # contains all the optim

        self.cali_data = torch.load(opt['cali_data'])
        
        
        print('a')
        print('a')
        print('a')
        print('a')
        print('a')
        print('a')
        print(self.device)
        print('a')
        print('a')
        print('a')
        print('a')
        print('a')
        

        self.net_F = build_network(opt["network_Q"])  # FP model
        self.load_network(
            self.net_F,
            self.opt["pathFP"]["pretrain_network_FP"],
            self.opt["pathFP"]["strict_load_FP"],
            "params",
        )
        self.net_F = self.net_F.to(self.device).eval()
        
        self.net_Q = build_network(opt["network_Q"])  # FP model
        self.load_network(
            self.net_Q,
            self.opt["pathFP"]["pretrain_network_FP"],
            self.opt["pathFP"]["strict_load_FP"],
            "params",
        )
        
        wq_params = {'n_bits': self.opt['bit'], 'channel_wise': True}
        aq_params = {'n_bits': self.opt['bit'], 'channel_wise': False}
        for name, module in self.net_Q.named_modules():
            if isinstance(module, WindowAttention):
                setattr(module, "matmul1", MatMul())
                setattr(module, "matmul2", MatMul())
                module.forward = MethodType(window_attention_forward, module)
                
        # self.build_quantized_network()  # Quantized model self.net_Q
        self.net_Q = quant_model(
            model = self.net_Q,
            input_quant_params=aq_params, weight_quant_params=wq_params
        )
        
        self.net_Q = self.net_Q.to(self.device)
        self.net_Q.eval()
        set_quant_state(self.net_Q, input_quant=True, weight_quant=True)
        # cali
        with torch.no_grad():
            print('Performing initial quantization ...')
            self.feed_data(self.cali_data)
            _ = self.net_Q(self.lq)
            print('initial quantization over ...')
        
        with torch.no_grad():
            module_dict={}
            q_model_slice = self.net_Q.layers
            for name, module in tqdm(q_model_slice.named_modules()):
                module_dict[name] = module
                idx = name.rfind('.')
                if idx == -1:
                    idx = 0
                father_name = name[:idx]
                if father_name in module_dict:
                    father_module = module_dict[father_name]
                else:
                    raise RuntimeError(f"father module {father_name} not found")

                if 'norm1' in name or 'norm2' in name or 'norm' in name:
                    if 'norm1' in name:
                        next_module = father_module.attn.qkv
                    elif 'norm2' in name:
                        next_module = father_module.mlp.fc1
                    else:
                        next_module = father_module.reduction
                    
                    act_delta = next_module.input_quantizer.delta.reshape(-1)
                    act_zero_point = next_module.input_quantizer.zero_point.reshape(-1)
                    act_min = -act_zero_point * act_delta
                    
                    target_delta = torch.mean(act_delta)
                    target_zero_point = torch.mean(act_zero_point)
                    target_min = -target_zero_point * target_delta

                    r = act_delta / target_delta
                    b = act_min / r - target_min

                    module.weight.data = module.weight.data / r
                    module.bias.data = module.bias.data / r - b

                    next_module.weight.data = next_module.weight.data * r
                    if next_module.bias is not None:

                        # new_b = torch.repeat_interleave(b, next_module.weight.data.size(1))
                        # mm = torch.mm(next_module.weight.data, new_b.reshape(-1,1)).reshape(-1)

                        next_module.bias.data = next_module.bias.data + torch.mm(next_module.weight.data, b.reshape(-1,1)).reshape(-1)
                        
                    else:
                        next_module.bias = Parameter(torch.Tensor(next_module.out_features))
                        next_module.bias.data = torch.mm(next_module.weight.data, b.reshape(-1,1)).reshape(-1)

                    next_module.input_quantizer.channel_wise = False
                    next_module.input_quantizer.delta = target_delta
                    next_module.input_quantizer.zero_point = target_zero_point
                    next_module.weight_quantizer.inited = False
                    
        set_quant_state(self.net_Q, input_quant=True, weight_quant=True)
        with torch.no_grad():
            _ = self.net_Q(self.lq)

    def setup_optimizers(self) -> None:
        pass


    def feed_data(self, data):
        self.lq = data["lq"].to(self.device)
        if "gt" in data:
            self.gt = data["gt"].to(self.device)



    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict["lq"] = self.lq.detach().cpu()
        out_dict["result"] = self.output.detach().cpu()
        if hasattr(self, "gt"):
            out_dict["gt"] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        """Save networks and training state."""
        self.save_network(self.net_Q, "net_Q", current_iter)
        self.save_training_state(epoch, current_iter)

    def validation(self, dataloader, current_iter, tb_logger, save_img=False):
        """Validation function.

        Args:
            dataloader (torch.utils.data.DataLoader): Validation dataloader.
            current_iter (int): Current iteration.
            tb_logger (tensorboard logger): Tensorboard logger.
            save_img (bool): Whether to save images. Default: False.
        """
        if self.opt["dist"]:
            self.dist_validation(dataloader, current_iter, tb_logger, save_img)
        else:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt["name"]
        with_metrics = self.opt["val"].get("metrics") is not None
        use_pbar = self.opt["val"].get("pbar", False)

        if with_metrics:
            if not hasattr(self, "metric_results"):  # only execute in the first run
                self.metric_results = {
                    metric: 0 for metric in self.opt["val"]["metrics"].keys()
                }
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit="image")
            
        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data["lq_path"][0]))[0]
            self.feed_data(val_data)
            self.test()
            
            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals["result"]])
            metric_data["img"] = sr_img
            if "gt" in visuals:
                gt_img = tensor2img([visuals["gt"]])
                metric_data["img2"] = gt_img
                del self.gt
            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt["val"]["metrics"].items():
                    # if name == 'psnr':
                    #     psnr = calculate_metric(metric_data, opt_)
                    self.metric_results[name] += calculate_metric(metric_data, opt_)
                    
            if save_img:
                if self.opt["is_train"]:
                    save_img_path = osp.join(
                        self.opt["path"]["visualization"],
                        img_name,
                        f"{img_name}_{current_iter}.png",
                    )
                else:
                    if self.opt["val"].get("suffix", False):
                        save_img_path = osp.join(
                            self.opt["path"]["visualization"],
                            dataset_name,
                            f'{img_name}_{self.opt["val"]["suffix"]}.png',
                        )
                    # else:
                    #     save_img_path = osp.join(
                    #         self.opt["path"]["visualization"],
                    #         dataset_name,
                    #         f'{img_name}_{psnr:.4f}_.png',
                    #     )
                imwrite(sr_img, save_img_path)
            
            if use_pbar:
                pbar.update(1)
                pbar.set_description(f"Test {img_name}")
        # self.time_forward = 0.0
        # self.time_get_visual = 0.0
        # self.time_save_img = 0.0
        # self.time_cal_metric = 0.0
        # self.time_total = 0.0
        
        # ,time_forward:{self.time_forward:.2f},time_get_visual:{self.time_get_visual:.2f}')
        if use_pbar:
            pbar.close()
        

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= idx + 1
                # update the best metric result
                self._update_best_metric_result(
                    dataset_name, metric, self.metric_results[metric], current_iter
                )

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)
        
        # print(f'time_forward:{self.time_forward:.2f}')
        # print(f'time_get_visual:{self.time_get_visual:.2f}')
        # print(f'time_save_img:{self.time_save_img:.2f}')
        # print(f'time_cal_metric:{self.time_cal_metric:.2f}')
        # print(f'time_total:{self.time_total:.2f}')


    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f"Validation {dataset_name}\n"
        for metric, value in self.metric_results.items():
            log_str += f"\t # {metric}: {value:.4f}"
            if hasattr(self, "best_metric_results"):
                log_str += (
                    f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                    f'{self.best_metric_results[dataset_name][metric]["iter"]} iter'
                )
            log_str += "\n"

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(
                    f"metrics/{dataset_name}/{metric}", value, current_iter
                )

    def test(self):
        # pad to multiplication of window_size
        if not self.opt['quant'].get('self_ensamble', False):
            
            window_size = self.opt["network_Q"]["window_size"]
            scale = self.opt.get("scale", 1)
            mod_pad_h, mod_pad_w = 0, 0
            _, _, h, w = self.lq.size()
            if h % window_size != 0:
                mod_pad_h = window_size - h % window_size
            if w % window_size != 0:
                mod_pad_w = window_size - w % window_size
            # img = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
            # 修改（按源码）
            img = self.lq
            img = torch.cat([img, torch.flip(img, [2])], 2)[:, :, : h + mod_pad_h, :]
            img = torch.cat([img, torch.flip(img, [3])], 3)[:, :, :, : w + mod_pad_w]
            
            if self.opt['quant'].get('bicubic', False):
                # print(img.size())
                # exit(0)
                resize = Resize((img.size(2) * self.opt['network_Q']['upscale'], img.size(3) * self.opt['network_Q']['upscale']))
                self.output = resize(img)
            elif self.opt['quant'].get('fp_test', False):
                self.net_F.eval()
                with torch.no_grad():
                    self.output = self.net_F(img)
            else:
                if hasattr(self, "net_g_ema"):
                    self.net_g_ema.eval()
                    with torch.no_grad():
                        self.output = self.net_g_ema(img)
                else:
                    self.net_Q.eval()
                    with torch.no_grad():
                        self.output = self.net_Q(img)
                    self.net_Q.eval()

            _, _, h, w = self.output.size()
            self.output = self.output[
                :, :, 0 : h - mod_pad_h * scale, 0 : w - mod_pad_w * scale
            ]
            # self.set_int_quant(False)
        else:
            transforms = [
                (lambda x: x, lambda y: y),  # No-op
                (lambda x: x.flip(2), lambda y: y.flip(2)),  # Vertical flip
                (lambda x: x.flip(3), lambda y: y.flip(3)),  # Horizontal flip
                (lambda x: x.flip(2).flip(3), lambda y: y.flip(2).flip(3)),  # Vertical + Horizontal flip
                (lambda x: x.transpose(2, 3), lambda y: y.transpose(2, 3)),  # Rotate 90 degrees
                (lambda x: x.transpose(2, 3).flip(2), lambda y: y.flip(2).transpose(2, 3)),  # Rotate 90 degrees + Vertical flip
                (lambda x: x.transpose(2, 3).flip(3), lambda y: y.flip(3).transpose(2, 3)),  # Rotate 90 degrees + Horizontal flip
                (lambda x: x.transpose(2, 3).flip(2).flip(3), lambda y: y.flip(2).flip(3).transpose(2, 3))  # Rotate 90 degrees + Vertical + Horizontal flip
            ]
            window_size = self.opt["network_Q"]["window_size"]
            scale = self.opt.get("scale", 1)
            mod_pad_h, mod_pad_w = 0, 0
            _, _, h, w = self.lq.size()
            if h % window_size != 0:
                mod_pad_h = window_size - h % window_size
            if w % window_size != 0:
                mod_pad_w = window_size - w % window_size
            # img = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
            # 修改（按源码）
            # real_outputs = []
            inputs = []
            
            for transform, inverse_transform in transforms:
                img = self.lq
                img = torch.cat([img, torch.flip(img, [2])], 2)[:, :, : h + mod_pad_h, :]
                img = torch.cat([img, torch.flip(img, [3])], 3)[:, :, :, : w + mod_pad_w]
                inputs.append(transform(img))
                # print(img.size())
            batch_input_1 = torch.cat(inputs[:4], dim=0)
            batch_input_2 = torch.cat(inputs[4:], dim=0)
            
            if hasattr(self, "net_g_ema"):
                self.net_g_ema.eval()
                with torch.no_grad():
                    output1 = self.net_g_ema(batch_input_1)
            else:
                self.net_Q.eval()
                with torch.no_grad():
                    output1 = self.net_Q(batch_input_1)
            if hasattr(self, "net_g_ema"):
                self.net_g_ema.eval()
                with torch.no_grad():
                    output2 = self.net_g_ema(batch_input_2)
            else:
                self.net_Q.eval()
                with torch.no_grad():
                    output2 = self.net_Q(batch_input_2)
                # real_outputs.append(output)
            # output = output1 + output2
            # output = torch.cat([output1, output2], dim=0)
            outputs1 = torch.chunk(output1, len(transforms), dim=0)
            outputs2 = torch.chunk(output2, len(transforms), dim=0)
            outputs = outputs1 + outputs2
            inverse_transformed_outputs = []
            for output, (_, inv_transform) in zip(outputs, transforms):
                inv_output = inv_transform(output)
                _, _, h_out, w_out = inv_output.size()
                cropped_output = inv_output[:, :, 0: h_out - mod_pad_h * scale, 0: w_out - mod_pad_w * scale]
                inverse_transformed_outputs.append(cropped_output)
                
            # Compute the mean of all the processed outputs
            final_output = torch.mean(torch.stack(inverse_transformed_outputs), dim=0)
            self.output = final_output
            return final_output
    
    
    def param_test_test(self):
        # pad to multiplication of window_size
        window_size = self.opt["network_FP"]["window_size"]
        scale = self.opt.get("scale", 1)
        mod_pad_h, mod_pad_w = 0, 0
        _, _, h, w = self.lq.size()
        if h % window_size != 0:
            mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            mod_pad_w = window_size - w % window_size
        # img = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        # 修改（按源码）
        img = self.lq
        img = torch.cat([img, torch.flip(img, [2])], 2)[:, :, : h + mod_pad_h, :]
        img = torch.cat([img, torch.flip(img, [3])], 3)[:, :, :, : w + mod_pad_w]
        
        with torch.no_grad():
            # self.output1 = self.net_F(img)
            self.output2 = self.net_Q(img)

        # _, _, h, w = self.output1.size()
        # self.output1 = self.output1[
        #     :, :, 0 : h - mod_pad_h * scale, 0 : w - mod_pad_w * scale
        # ]
        _, _, h, w = self.output2.size()
        self.output2 = self.output2[
            :, :, 0 : h - mod_pad_h * scale, 0 : w - mod_pad_w * scale
        ]
        # self.set_int_quant(False)

    def set_int_quant(self, enable: bool):
        for f in self.quant_linears.values():
            f: QuantLinear
            f.weight_quantizer.set_int_quant(enable)
            f.act_quantizer.set_int_quant(enable)

    def _initialize_best_metric_results(self, dataset_name):
        """Initialize the best metric results dict for recording the best metric value and iteration."""
        if (
            hasattr(self, "best_metric_results")
            and dataset_name in self.best_metric_results
        ):
            return
        elif not hasattr(self, "best_metric_results"):
            self.best_metric_results = dict()

        # add a dataset record
        record = dict()
        for metric, content in self.opt["val"]["metrics"].items():
            better = content.get("better", "higher")
            init_val = float("-inf") if better == "higher" else float("inf")
            record[metric] = dict(better=better, val=init_val, iter=-1)
        self.best_metric_results[dataset_name] = record

    def _update_best_metric_result(self, dataset_name, metric, val, current_iter):
        if self.best_metric_results[dataset_name][metric]["better"] == "higher":
            if val >= self.best_metric_results[dataset_name][metric]["val"]:
                self.best_metric_results[dataset_name][metric]["val"] = val
                self.best_metric_results[dataset_name][metric]["iter"] = current_iter
        else:
            if val <= self.best_metric_results[dataset_name][metric]["val"]:
                self.best_metric_results[dataset_name][metric]["val"] = val
                self.best_metric_results[dataset_name][metric]["iter"] = current_iter

    def model_ema(self, decay=0.999):
        net_g = self.get_bare_model(self.net_g)

        net_g_params = dict(net_g.named_parameters())
        net_g_ema_params = dict(self.net_g_ema.named_parameters())

        for k in net_g_ema_params.keys():
            net_g_ema_params[k].data.mul_(decay).add_(
                net_g_params[k].data, alpha=1 - decay
            )

    def get_current_log(self):
        # for f in self.quant_linears.values():
        #     f: QuantLinear
        #     f.w
        bits_weight = []
        bits_act = []
        size_total_weight = 0
        size_total_act = 0
        for name, module in self.net_Q.named_modules():
            if isinstance(module, QuantLinear):
                self.log_dict[f"param_{name}_act_lb"] = float(
                    module.act_quantizer.lower_bound
                )
                self.log_dict[f"param_{name}_act_ub"] = float(
                    module.act_quantizer.upper_bound
                )
                self.log_dict[f"param_{name}_act_n_bit"] = float(
                    module.act_quantizer.n_bit
                )

                self.log_dict[f"param_{name}_weight_lb"] = float(
                    module.weight_quantizer.lower_bound
                )
                self.log_dict[f"param_{name}_weight_ub"] = float(
                    module.weight_quantizer.upper_bound
                )
                self.log_dict[f"param_{name}_weight_n_bit"] = float(
                    module.weight_quantizer.n_bit
                )
                size_total_weight += module.weight_quantizer.size_of_input
                size_total_act += module.act_quantizer.size_of_input
                
                bits_weight.append(float(module.weight_quantizer.n_bit * module.weight_quantizer.size_of_input))
                bits_act.append(float(module.act_quantizer.n_bit * module.act_quantizer.size_of_input))
                
                
        self.log_dict['param_weight_avg_n_bit'] = np.sum(bits_weight) / size_total_weight
        self.log_dict['param_act_avg_n_bit'] = np.sum(bits_act) / size_total_act

        return self.log_dict

    def model_to_device(self, net):
        """Model to device. It also warps models with DistributedDataParallel
        or DataParallel.

        Args:
            net (nn.Module)
        """
        print('a')
        print('a')
        print('a')
        print('a')
        print('a')
        print('a')
        print('a')
        print('a')
        net = net.to(self.device)
        if self.opt["dist"]:
            find_unused_parameters = self.opt.get("find_unused_parameters", False)
            net = DistributedDataParallel(
                net,
                device_ids=[torch.cuda.current_device()],
                find_unused_parameters=find_unused_parameters,
            )
        elif self.opt["num_gpu"] > 1:
            net = DataParallel(net)
        return net

    def get_optimizer(self, optim_type, params, lr, **kwargs) -> Adam:
        if optim_type == "Adam":
            optimizer = torch.optim.Adam(params, lr, **kwargs)
        else:
            raise NotImplementedError(f"optimizer {optim_type} is not supperted yet.")
        return optimizer

    def setup_schedulers(self):
        pass

    def get_bare_model(self, net):
        """Get bare model, especially under wrapping with
        DistributedDataParallel or DataParallel.
        """
        if isinstance(net, (DataParallel, DistributedDataParallel)):
            net = net.module
        return net

    @master_only
    def print_network(self, net):
        """Print the str and parameter number of a network.

        Args:
            net (nn.Module)
        """
        if isinstance(net, (DataParallel, DistributedDataParallel)):
            net_cls_str = f"{net.__class__.__name__} - {net.module.__class__.__name__}"
        else:
            net_cls_str = f"{net.__class__.__name__}"

        net = self.get_bare_model(net)
        net_str = str(net)
        net_params = sum(map(lambda x: x.numel(), net.parameters()))

        logger = get_root_logger()
        logger.info(f"Network: {net_cls_str}, with parameters: {net_params:,d}")
        logger.info(net_str)

    def _set_lr(self, lr_groups_l):
        """Set learning rate for warmup.

        Args:
            lr_groups_l (list): List for lr_groups, each for an optimizer.
        """
        for optimizer, lr_groups in zip(self.optimizers, lr_groups_l):
            for param_group, lr in zip(optimizer.param_groups, lr_groups):
                param_group["lr"] = lr

    def _get_init_lr(self):
        """Get the initial lr, which is set by the scheduler."""
        init_lr_groups_l = []
        for optimizer in self.optimizers:
            init_lr_groups_l.append([v["initial_lr"] for v in optimizer.param_groups])
        return init_lr_groups_l

    def update_learning_rate(self, current_iter, warmup_iter=-1):
        """Update learning rate.

        Args:
            current_iter (int): Current iteration.
            warmup_iter (int)： Warmup iter numbers. -1 for no warmup.
                Default： -1.
        """
        if current_iter > 1:
            for scheduler in self.schedulers:
                scheduler.step()
        # set up warm-up learning rate
        if current_iter < warmup_iter:
            # get initial lr for each group
            init_lr_g_l = self._get_init_lr()
            # modify warming-up learning rates
            # currently only support linearly warm up
            warm_up_lr_l = []
            for init_lr_g in init_lr_g_l:
                warm_up_lr_l.append([v / warmup_iter * current_iter for v in init_lr_g])
            # set learning rate
            self._set_lr(warm_up_lr_l)

    def get_current_learning_rate(self):
        return [param_group["lr"] for param_group in self.optimizers[0].param_groups]

    @master_only
    def save_network(self, net, net_label, current_iter, param_key="params"):
        """Save networks.

        Args:
            net (nn.Module | list[nn.Module]): Network(s) to be saved.
            net_label (str): Network label.
            current_iter (int): Current iter number.
            param_key (str | list[str]): The parameter key(s) to save network.
                Default: 'params'.
        """
        if current_iter == -1:
            current_iter = "latest"
        save_filename = f"{net_label}_{current_iter}.pth"
        save_path = os.path.join(self.opt["path"]["models"], save_filename)

        net = net if isinstance(net, list) else [net]
        param_key = param_key if isinstance(param_key, list) else [param_key]
        assert len(net) == len(
            param_key
        ), "The lengths of net and param_key should be the same."

        save_dict = {}
        for net_, param_key_ in zip(net, param_key):
            net_ = self.get_bare_model(net_)
            state_dict = net_.state_dict()
            for key, param in state_dict.items():
                if key.startswith("module."):  # remove unnecessary 'module.'
                    key = key[7:]
                state_dict[key] = param.cpu()
            save_dict[param_key_] = state_dict

        # avoid occasional writing errors
        retry = 3
        while retry > 0:
            try:
                torch.save(save_dict, save_path)
            except Exception as e:
                logger = get_root_logger()
                logger.warning(
                    f"Save model error: {e}, remaining retry times: {retry - 1}"
                )
                time.sleep(1)
            else:
                break
            finally:
                retry -= 1
        if retry == 0:
            logger.warning(f"Still cannot save {save_path}. Just ignore it.")
            # raise IOError(f'Cannot save {save_path}.')

    def _print_different_keys_loading(self, crt_net, load_net, strict=True):
        """Print keys with different name or different size when loading models.

        1. Print keys with different names.
        2. If strict=False, print the same key but with different tensor size.
            It also ignore these keys with different sizes (not load).

        Args:
            crt_net (torch model): Current network.
            load_net (dict): Loaded network.
            strict (bool): Whether strictly loaded. Default: True.
        """
        crt_net = self.get_bare_model(crt_net)
        crt_net = crt_net.state_dict()
        crt_net_keys = set(crt_net.keys())
        load_net_keys = set(load_net.keys())

        logger = get_root_logger()
        if crt_net_keys != load_net_keys:
            logger.warning("Current net - loaded net:")
            for v in sorted(list(crt_net_keys - load_net_keys)):
                logger.warning(f"  {v}")
            logger.warning("Loaded net - current net:")
            for v in sorted(list(load_net_keys - crt_net_keys)):
                logger.warning(f"  {v}")

        # check the size for the same keys
        if not strict:
            common_keys = crt_net_keys & load_net_keys
            for k in common_keys:
                if crt_net[k].size() != load_net[k].size():
                    logger.warning(
                        f"Size different, ignore [{k}]: crt_net: "
                        f"{crt_net[k].shape}; load_net: {load_net[k].shape}"
                    )
                    load_net[k + ".ignore"] = load_net.pop(k)

    def load_network(self, net, load_path, strict=True, param_key="params"):
        """Load network.

        Args:
            load_path (str): The path of networks to be loaded.
            net (nn.Module): Network.
            strict (bool): Whether strictly loaded.
            param_key (str): The parameter key of loaded network. If set to
                None, use the root 'path'.
                Default: 'params'.
        """
        logger = get_root_logger()
        net = self.get_bare_model(net)
        load_net = torch.load(load_path, map_location=lambda storage, loc: storage)
        if param_key is not None:
            if param_key not in load_net and "params" in load_net:
                param_key = "params"
                logger.info("Loading: params_ema does not exist, use params.")
            load_net = load_net[param_key]
        logger.info(
            f"Loading {net.__class__.__name__} model from {load_path}, with param key: [{param_key}]."
        )
        # remove unnecessary 'module.'
        for k, v in deepcopy(load_net).items():
            if k.startswith("module."):
                load_net[k[7:]] = v
                load_net.pop(k)
        self._print_different_keys_loading(net, load_net, strict)
        net.load_state_dict(load_net, strict=strict)

    @master_only
    def save_training_state(self, epoch, current_iter):
        """Save training states during training, which will be used for
        resuming.

        Args:
            epoch (int): Current epoch.
            current_iter (int): Current iteration.
        """
        if current_iter != -1:
            state = {
                "epoch": epoch,
                "iter": current_iter,
                "optimizers": [],
                "schedulers": [],
            }
            for o in self.optimizers:
                state["optimizers"].append(o.state_dict())
            for s in self.schedulers:
                state["schedulers"].append(s.state_dict())
            save_filename = f"{current_iter}.state"
            save_path = os.path.join(self.opt["path"]["training_states"], save_filename)

            # avoid occasional writing errors
            retry = 3
            while retry > 0:
                try:
                    torch.save(state, save_path)
                except Exception as e:
                    logger = get_root_logger()
                    logger.warning(
                        f"Save training state error: {e}, remaining retry times: {retry - 1}"
                    )
                    time.sleep(1)
                else:
                    break
                finally:
                    retry -= 1
            if retry == 0:
                logger.warning(f"Still cannot save {save_path}. Just ignore it.")
                # raise IOError(f'Cannot save {save_path}.')

    def resume_training(self, resume_state):
        """Reload the optimizers and schedulers for resumed training.

        Args:
            resume_state (dict): Resume state.
        """
        resume_optimizers = resume_state["optimizers"]
        resume_schedulers = resume_state["schedulers"]
        assert len(resume_optimizers) == len(
            self.optimizers
        ), "Wrong lengths of optimizers"
        assert len(resume_schedulers) == len(
            self.schedulers
        ), "Wrong lengths of schedulers"
        for i, o in enumerate(resume_optimizers):
            self.optimizers[i].load_state_dict(o)
        for i, s in enumerate(resume_schedulers):
            self.schedulers[i].load_state_dict(s)

    def reduce_loss_dict(self, loss_dict):
        """reduce loss dict.

        In distributed training, it averages the losses among different GPUs .

        Args:
            loss_dict (OrderedDict): Loss dict.
        """
        with torch.no_grad():
            if self.opt["dist"]:
                keys = []
                losses = []
                for name, value in loss_dict.items():
                    keys.append(name)
                    losses.append(value)
                losses = torch.stack(losses, 0)
                torch.distributed.reduce(losses, dst=0)
                if self.opt["rank"] == 0:
                    losses /= self.opt["world_size"]
                loss_dict = {key: loss for key, loss in zip(keys, losses)}

            log_dict = OrderedDict()
            for name, value in loss_dict.items():
                log_dict[name] = value.mean().item()

            return log_dict



