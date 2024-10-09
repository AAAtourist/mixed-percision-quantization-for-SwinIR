import logging
import torch
from os import path as osp
import os

from basicsr.data import build_dataloader, build_dataset
from basicsr.models import build_model
from basicsr.models.sr_model import List_Quantizers, QuantLinear
from basicsr.utils import get_env_info, get_root_logger, get_time_str, make_exp_dirs
from basicsr.utils.options import dict2str, parse_options

def test_pipeline(root_path):
    # parse options, set distributed setting, set ramdom seed
    opt, _ = parse_options(root_path, is_train=False)
    os.environ['CUDA_VISIBLE_DEVICES'] = f"{opt['gpu']}"

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # mkdir and initialize loggers
    make_exp_dirs(opt)
    log_file = osp.join(opt['path']['log'], f"test_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))

    # create test dataset and dataloader
    test_loaders = []
    for _, dataset_opt in sorted(opt['datasets'].items()):
        test_set = build_dataset(dataset_opt)
        test_loader = build_dataloader(
            test_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None, seed=opt['manual_seed'])
        logger.info(f"Number of test images in {dataset_opt['name']}: {len(test_set)}")
        test_loaders.append(test_loader)

    

    # create model
    model = build_model(opt)
    n_l, n_m = 0, 0
    for _, module in model.net_g.named_modules():
        if not isinstance(module, List_Quantizers):
            continue
        if isinstance(module.quantizers_dict["best_module"], QuantLinear): #26, 53, 66, 70, 76, 77, 80, 89, 93, 95
            if n_l in [53, 66, 70, 76, 77, 80]:
                module.change_bit('uni', 2)
            n_l += 1
            

    '''type_ = opt['quantization']['type']
    is_uni = 'uni' if opt['uni'] else 'log'
    number = opt['quantization']['number']
    bit_ = opt['quantization']['bit']
    n = 0

    for _, module in model.net_g.named_modules():
        if not isinstance(module, List_Quantizers):
            continue
        if isinstance(module.quantizers_dict["best_module"], QuantLinear):
            if type_ == 'linear':
                if n == number:
                    print(f'linear{n}')
                    module.change_bit(is_uni, bit_)
                n += 1
        elif type_ == 'matmul':
            if n == number:
                print(f'matmul{n}')
                module.change_bit(is_uni, bit_)
            n += 1'''



    for test_loader in test_loaders:
        test_set_name = test_loader.dataset.opt['name']
        logger.info(f'Testing {test_set_name}...')
        model.validation(test_loader, current_iter=opt['name'], tb_logger=None, save_img=opt['val']['save_img'])


if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    test_pipeline(root_path)
