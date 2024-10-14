import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew
import os

def generate_and_save_histogram(selected_data, ss):
    # 将数据从 tensor 转换为 numpy 数组 
    selected_data = selected_data.cpu().numpy()
    
    global num_plot
    if 'num_plot' not in globals():
        num_plot = 0 

    #skew_value = skew(selected_data.flatten())
    #save_path = f'/data/user/tourist/mixed-percision-quantization-for-SwinIR/draw_PDF/input_qkv/photo{num_plot // (ss * 24) + 1},RSTB{(num_plot // (ss * 6)) % 4},SwinTB{(num_plot // (ss)) % 6}'
    #save_path = f'/data/user/tourist/mixed-percision-quantization-for-SwinIR/draw_PDF/matrix_v/photo{num_plot // 1440 + 1},RSTB{(num_plot // 360) % 4},SwinTB{(num_plot // 60) % 6}'

    save_path = f'/data/user/tourist/mixed-percision-quantization-for-SwinIR/draw_PDF/input_qkv/linear{num_plot // ss}, '
    os.makedirs(save_path, exist_ok=True)
    
    file_name = f'channel{num_plot % ss}.png'
    #file_name = f'photo{num_plot // 1440 + 1},RSTB{(num_plot // 360) % 4},SwinTB{(num_plot // 60) % 6},{num_plot % 60}.png'
    #if ((num_plot // (ss * 6)) % 4) == 0:

    min_value, max_value = selected_data.min(), selected_data.max()

    fig, ax = plt.subplots(figsize=(5, 4))
    
    ax.hist(selected_data.flatten(), bins=200, density=False, alpha=0.6, color='b')
    ax.set_title(f'sample {num_plot}, range = [{float(min_value):.3f}, {float(max_value):.3f}]')

    ax.text(0.05, 0.8, f'min={min_value:.2f}', transform=ax.transAxes, color='red')
    ax.text(0.05, 0.7, f'max={max_value:.2f}', transform=ax.transAxes, color='green')

    # 调整布局
    plt.tight_layout()
    num_plot += 1

    # 生成文件名并保存图像
    file_path = os.path.join(save_path, file_name)
    plt.savefig(file_path, dpi=200)
    
    plt.close(fig)

def draw_pdf(x):

    sampling_probability = 1
    data = torch.abs(x)
    ss = data.shape[2]
    num_channels = data.shape[2]
    num_sampled_channels = int(num_channels * sampling_probability)

    selected_channels = np.random.choice(num_channels, num_sampled_channels, replace=False) 
    #save_path = '/data/user/tourist/mixed-percision-quantization-for-SwinIR/draw_PDF/input_qkv'
    #os.makedirs(save_path, exist_ok=True)

    for channel in range(num_channels):
        channel_data = data[:,:, channel]
        generate_and_save_histogram(channel_data, ss)
    
    #generate_and_save_histogram(data, save_path)
