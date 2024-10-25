
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import wandb

def draw_plot(weight):

    global num
    if 'num' not in globals(): 
        num = 0 
    
    weight = torch.abs(weight).detach()
    if weight.is_cuda:
        weight = weight.cpu().numpy()
    else:
        weight = weight.numpy()
    '''dic = ['qkv_linear', 'attn_proj', 'mlp_fc1', 'mlp_fc2']
    name = dic[num % 4]
    '''



    input_channels = weight.shape[0]#60
    output_channels = weight.shape[1]#180

    X = np.arange(output_channels)
    Y = np.arange(input_channels)
    X, Y = np.meshgrid(X, Y)

    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111, projection='3d')

    top_1_percent_threshold = np.percentile(weight, 99)
    # 创建颜色数组，初始化为蓝色，shape 为 (n, 4) 表示 RGBA 值
    colors = np.zeros((weight.size, 4))

    colors[:] = [0, 0, 1, 0.3]  # 蓝色

    top_1_indices = weight.ravel() >= top_1_percent_threshold
    colors[top_1_indices] = [1, 0, 0, 0.8]  # 红色

    ax.bar3d(X.ravel(), Y.ravel(), np.zeros_like(X.ravel()), 0.2, 0.2, weight.ravel(), color=colors)

    ax.set_xlabel('Channel')
    ax.set_ylabel('')
    ax.set_zlabel('Magnitude')

    ax.grid(True)
    ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax.tick_params(axis='y', which='both', bottom=False, top=False, labelleft=False)



    ax.text2D(0.05, 0.95, "X", transform=ax.transAxes, fontsize=15, weight='bold')
    plt.title(f'input_sampled{num}', fontsize=12)
    wandb.log({f"3D_plot{num}": wandb.Image(plt)})
    #plt.show()

    #save_path = os.path.join(save_folder, f'input_sampled{num}.png')
    num += 1

    plt.close(fig)


def draw_3d_plot(x):

    sampling_probability = 0.01
    data = x
    #data = torch.max(x, dim=0)[0]
    num_channels = data.shape[-1]
    batch = data.shape[0]
    num_sampled = int(batch * sampling_probability)

    selected_channels = np.random.choice(batch, num_sampled, replace=False) 
    #save_path = '/data/user/tourist/mixed-percision-quantization-for-SwinIR/draw_PDF/input_qkv'
    #os.makedirs(save_path, exist_ok=True)
    #draw_plot(data)

    for batch in selected_channels:
        channel_data = data[batch,:,:]
        draw_plot(channel_data)
