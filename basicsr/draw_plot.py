import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.cm as cm


def draw_plot(x):
    global num_plot
    if 'num_plot' not in globals(): 
        num_plot = 0 
    save_directory = '/data/user/tourist/mixed-percision-quantization-for-SwinIR/boxplot/attn_after_fc1'
    os.makedirs(save_directory, exist_ok=True)

    draw_data = x.reshape(-1, 120).cpu().numpy()[:, 60:]

    colors = cm.viridis(np.linspace(0, 1, 60))

    plt.figure(figsize=(12, 6))
    box = plt.boxplot([draw_data[:, j] for j in range(0, draw_data.shape[1])], patch_artist=True, showfliers=False)
    
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)

    #positions = np.arange(61, 121)
    #plt.xticks(ticks=positions[::5], labels=positions[::5])

    plt.xticks(ticks=range(1, 61, 5), labels=range(61, 121, 5))

    plt.xlabel('Channel Index')
    plt.ylabel('Range')

    file_name = f'phtot{num_plot // 24},RSTB{(num_plot // 6) % 4},SwinTB{num_plot % 6 + 1}.png'
    num_plot += 1
    file_path = os.path.join(save_directory, file_name)

    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    
    plt.close()
