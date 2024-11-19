import itertools
from typing import Tuple
import torch
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm
from torch.nn import Module
from torch import Tensor

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.quantize import QuantLinear, quant_model, QuantMatMul
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
from basicsr.smooth_networks import smooth_network

import warnings

@MODEL_REGISTRY.register()
class SRModel(BaseModel):
    """Base SR model for single image super-resolution."""

    def __init__(self, opt):
        super(SRModel, self).__init__(opt)

        # define network
        self.net_F = build_network(opt["network_Q"])
        self.net_F = self.model_to_device(self.net_F)
        #self.print_network(self.net_F)

        # load pretrained models
        load_path = self.opt['pathFP'].get('pretrain_network_FP', None)
        if load_path is not None:
            param_key = self.opt['pathFP'].get('param_key_FP', 'params')
            self.load_network(self.net_F, load_path, self.opt['pathFP'].get('strict_load_FP', True), param_key)

        self.net_Q = build_network(opt["network_Q"])
        self.net_Q = self.model_to_device(self.net_Q)
        #self.print_network(self.net_F)

        # load pretrained models
        load_path = self.opt['pathFP'].get('pretrain_network_FP', None)
        if load_path is not None:
            param_key = self.opt['pathFP'].get('param_key_FP', 'params')
            self.load_network(self.net_Q, load_path, self.opt['pathFP'].get('strict_load_FP', True), param_key)

        #torch.autograd.set_detect_anomaly(True)
        self.net_Q.smooth_network = smooth_network(opt['clusters'], opt['network_Q']['embed_dim'])
        #_ = get_global_smooth_network(opt['clusters'], opt['network_Q']['embed_dim'])
        self.net_Q = quant_model(
            model = self.net_Q,
            quant_params=self.opt['quantization']
            )
        
        self.net_Q = self.model_to_device(self.net_Q)
        self.cali_data = torch.load(opt['cali_data'])
        self.net_Q.eval()
        '''
        path = "/data/user/tourist/mixed-percision-quantization-for-SwinIR/pretrained_model/SwinIR_x2_cali_done.pth"
        self.net_Q = torch.load(path)
        self.net_Q = self.model_to_device(self.net_Q)
        '''
        with torch.no_grad():
            self.feed_data(self.cali_data)
            _ = self.net_Q(self.lq)
            #torch.save(self.net_Q, "/data/user/tourist/mixed-percision-quantization-for-SwinIR/pretrained_model/SwinIR_x2_cali_done.pth")
        '''
            state_dict = self.net_Q.state_dict()
            for key, _ in state_dict.items():
                print(key)
        '''
        if self.opt['path']['pretrain_network_Q'] != None:
            self.load_network(
                self.net_Q,
                self.opt["path"]["pretrain_network_Q"],
                self.opt["path"]["strict_load_Q"],
                "params",
            )
            print('load')
            for name, module in self.net_Q.named_modules():
                if isinstance(module, (QuantLinear, QuantMatMul)):
                    module.first_time = False

                    
        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_F.eval()
        self.net_Q.eval()
        train_opt = self.opt['train']

        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('feature_loss'):
            self.feature_loss = build_loss(train_opt['feature_loss']).to(self.device)
        else:
            self.feature_loss = None

        if train_opt.get('smooth_loss'):
            #self.Smooth_loss = True
            self.Smooth_loss = build_loss(train_opt['smooth_loss']).to(self.device)

        if self.cri_pix is None and self.feature_loss is None:
            raise ValueError('Both pixel and feature_loss are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()
        self.build_hooks_on_Q_and_F()
        self.build_hooks_on_smooth_networks()
        
    def setup_optimizers(self):
        from basicsr.quantize import QuantLinear, qkv_module, QuantMatMul

        train_opt = self.opt['train']
        optim_bound_matrix_params = []
        logger = get_root_logger()

        for name, module in self.net_Q.named_modules():
            if isinstance(module, (QuantLinear, QuantMatMul)):

                optim_bound_matrix_params.extend(module.get_bound_param())
                logger.info(f'{name} is added in optim_bound_params')

        net = smooth_network()
        
        optim_bound_matrix_params.extend(net.A_matrices)
        optim_bound_matrix_params.extend(net.B_matrices)

        logger.info('global_smooth_network is added in optim_matrix_params')
                
        optim_type = train_opt['optim_matrix_params'].pop('type')
        self.optimizer_bound_matrix = self.get_optimizer(optim_type, optim_bound_matrix_params, **train_opt['optim_matrix_params'])
        self.optimizers.append(self.optimizer_bound_matrix)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    def build_hooks_on_Q_and_F(self):
        from basicsr.archs.swinir_arch import BasicLayer, SwinTransformerBlock

        self.feature_F = []
        self.feature_Q = []

        if self.opt["quant"]["hook_per_layer"]:
            hook_type = BasicLayer
        elif self.opt["quant"]["hook_per_block"]:
            hook_type = SwinTransformerBlock

        def hook_layer_forward(
            module: Module, input: Tensor, output: Tensor, buffer: list
        ):
            buffer.append(output)

        for name, module in self.net_F.named_modules():
            if isinstance(module, hook_type):
                module.register_forward_hook(
                    partial(hook_layer_forward, buffer=self.feature_F)
                )
        for name, module in self.net_Q.named_modules():
            if isinstance(module, hook_type):
                module.register_forward_hook(
                    partial(hook_layer_forward, buffer=self.feature_Q)
                )

    def build_hooks_on_smooth_networks(self):

        self.smooth_feature = []
        def hook_smooth_network_loss(
            module: Module, input: Tensor, output: Tuple, buffer: list
        ):
            buffer.append(output[0])
            buffer.append(output[1])

        net = smooth_network()
        net.register_forward_hook(
                    partial(hook_smooth_network_loss, buffer=self.smooth_feature)
                )

        '''for name, module in self.net_Q.named_modules():
            if isinstance(module, QuantLinear) and module.smooth_network is not None:
                module.smooth_network.register_forward_hook(
                    partial(hook_smooth_network_loss, buffer=self.smooth_loss)
                )'''
                
    def optimize_parameters(self, current_iter):
        self.feature_Q.clear()
        self.feature_F.clear()
        self.smooth_feature.clear()

        self.output_Q = self.net_Q(self.lq)
        with torch.no_grad():
            self.output_F = self.net_F(self.lq)

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        #current_iter = (1 + current_iter) // 100
        #if current_iter % 2 == 1:
        if self.cri_pix:
            l_pix = self.cri_pix(self.output_Q, self.output_F) / self.output_Q.numel() * self.output_Q.size(0)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix
            
            #print('l_pix', l_pix)


        # feature loss
        if self.feature_loss:
            l_feature = 0
            idx = 0
            for feature_q, feature_f in zip(self.feature_Q, self.feature_F):
                norm_q = torch.norm(feature_q, dim=(1, 2)).detach()
                norm_f = torch.norm(feature_f, dim=(1, 2)).detach()

                norm_q.unsqueeze_(1).unsqueeze_(2)
                norm_f.unsqueeze_(1).unsqueeze_(2)

                feature_q = feature_q / norm_q
                feature_f = feature_f / norm_f

                fi = self.feature_loss(feature_q, feature_f) / feature_q.numel()

                loss_dict[f'l_feature_{idx}'] = fi
                l_feature += fi
                idx += 1
            
            #loss_dict['l_feature'] = l_feature
            l_total += l_feature

        #self.optimizer_bound.zero_grad()
        #l_total.backward()
        #self.optimizer_bound.step()

        #if current_iter % 2 == 0:
        if self.Smooth_loss:
            l_smooth = 0
            idx = 0
            for feature_A, feature_B in zip(self.smooth_feature[::2], self.smooth_feature[1::2]):
                fi = self.Smooth_loss(feature_A) + self.Smooth_loss(feature_B)
                fi = fi / len(self.smooth_feature)
                #loss_dict[f'smooth_feature_{idx}'] = fi
                l_smooth += fi
                idx += 1

            loss_dict['smooth_feature'] = l_smooth

        l_total += l_smooth

        loss_dict['orth_loss'] = self.net_Q.smooth_network.orth_loss()

        self.optimizer_bound_matrix.zero_grad()
        l_total.backward()
        self.optimizer_bound_matrix.step()

        
        self.log_dict = self.reduce_loss_dict(loss_dict)

    def test(self):
        # pad to multiplication of window_size
        # we do not use self-ensamble.
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
        """Save networks and training state."""
        self.save_network(self.net_Q, "net_Q", current_iter)
        self.save_training_state(epoch, current_iter)
