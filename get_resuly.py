import re
import os

file_dirs_minmaxquant = [
    r'/data/user/tourist/mixed-percision-quantization-for-SwinIR/results/test_SwinIR_light_x2_bit2_percentile',
    r'/data/user/tourist/mixed-percision-quantization-for-SwinIR/results/test_SwinIR_light_x2_bit3_percentile',
    r'/data/user/tourist/mixed-percision-quantization-for-SwinIR/results/test_SwinIR_light_x2_bit4_percentile',
    r'/data/user/tourist/mixed-percision-quantization-for-SwinIR/results/test_SwinIR_light_x3_bit2_percentile',
    r'/data/user/tourist/mixed-percision-quantization-for-SwinIR/results/test_SwinIR_light_x3_bit3_percentile',
    r'/data/user/tourist/mixed-percision-quantization-for-SwinIR/results/test_SwinIR_light_x3_bit4_percentile',
    r'/data/user/tourist/mixed-percision-quantization-for-SwinIR/results/test_SwinIR_light_x4_bit2_percentile',
    r'/data/user/tourist/mixed-percision-quantization-for-SwinIR/results/test_SwinIR_light_x4_bit3_percentile',
    r'/data/user/tourist/mixed-percision-quantization-for-SwinIR/results/test_SwinIR_light_x4_bit4_percentile',

    r'/data/user/tourist/mixed-percision-quantization-for-SwinIR/results/test_SwinIR_light_x2_bit2_token',
    r'/data/user/tourist/mixed-percision-quantization-for-SwinIR/results/test_SwinIR_light_x2_bit3_token',
    r'/data/user/tourist/mixed-percision-quantization-for-SwinIR/results/test_SwinIR_light_x2_bit4_token',
    r'/data/user/tourist/mixed-percision-quantization-for-SwinIR/results/test_SwinIR_light_x3_bit2_token',
    r'/data/user/tourist/mixed-percision-quantization-for-SwinIR/results/test_SwinIR_light_x3_bit3_token',
    r'/data/user/tourist/mixed-percision-quantization-for-SwinIR/results/test_SwinIR_light_x3_bit4_token',
    r'/data/user/tourist/mixed-percision-quantization-for-SwinIR/results/test_SwinIR_light_x4_bit2_token',
    r'/data/user/tourist/mixed-percision-quantization-for-SwinIR/results/test_SwinIR_light_x4_bit3_token',
    r'/data/user/tourist/mixed-percision-quantization-for-SwinIR/results/test_SwinIR_light_x4_bit4_token',

    r'/data/user/tourist/mixed-percision-quantization-for-SwinIR/results/test_SwinIR_light_x2_bit2_percentile_token',
    r'/data/user/tourist/mixed-percision-quantization-for-SwinIR/results/test_SwinIR_light_x2_bit3_percentile_token',
    r'/data/user/tourist/mixed-percision-quantization-for-SwinIR/results/test_SwinIR_light_x2_bit4_percentile_token',
    r'/data/user/tourist/mixed-percision-quantization-for-SwinIR/results/test_SwinIR_light_x3_bit2_percentile_token',
    r'/data/user/tourist/mixed-percision-quantization-for-SwinIR/results/test_SwinIR_light_x3_bit3_percentile_token',
    r'/data/user/tourist/mixed-percision-quantization-for-SwinIR/results/test_SwinIR_light_x3_bit4_percentile_token',
    r'/data/user/tourist/mixed-percision-quantization-for-SwinIR/results/test_SwinIR_light_x4_bit2_percentile_token',
    r'/data/user/tourist/mixed-percision-quantization-for-SwinIR/results/test_SwinIR_light_x4_bit3_percentile_token',
    r'/data/user/tourist/mixed-percision-quantization-for-SwinIR/results/test_SwinIR_light_x4_bit4_percentile_token',
]
res_file_minmaxquant = r'/data/user/tourist/mixed-percision-quantization-for-SwinIR/results/minmaxquant_res.txt'

file_dirs_RepQ = [
    r'D:\vscode-files\PTQ-0803\results\test_repQ_x2_bit4',
    r'D:\vscode-files\PTQ-0803\results\test_repQ_x2_bit3',
    r'D:\vscode-files\PTQ-0803\results\test_repQ_x2_bit2',
    r'D:\vscode-files\PTQ-0803\results\test_repQ_x3_bit4',
    r'D:\vscode-files\PTQ-0803\results\test_repQ_x3_bit3',
    r'D:\vscode-files\PTQ-0803\results\test_repQ_x3_bit2',
    r'D:\vscode-files\PTQ-0803\results\test_repQ_x4_bit4',
    r'D:\vscode-files\PTQ-0803\results\test_repQ_x4_bit3',
    r'D:\vscode-files\PTQ-0803\results\test_repQ_x4_bit2',

]
res_file_RepQ = r'D:\vscode-files\PTQ-0803\stat_res\RepQ_res_bs32.txt'

file_dirs_fqvit = [
    r'D:\vscode-files\PTQ-0803\results\test_fqvit_x2_bit8',
    r'D:\vscode-files\PTQ-0803\results\test_fqvit_x2_bit4',
    r'D:\vscode-files\PTQ-0803\results\test_fqvit_x2_bit3',
    r'D:\vscode-files\PTQ-0803\results\test_fqvit_x2_bit2',
    r'D:\vscode-files\PTQ-0803\results\test_fqvit_x3_bit8',
    r'D:\vscode-files\PTQ-0803\results\test_fqvit_x3_bit4',
    r'D:\vscode-files\PTQ-0803\results\test_fqvit_x3_bit3',
    r'D:\vscode-files\PTQ-0803\results\test_fqvit_x3_bit2',
    r'D:\vscode-files\PTQ-0803\results\test_fqvit_x4_bit8',
    r'D:\vscode-files\PTQ-0803\results\test_fqvit_x4_bit4',
    r'D:\vscode-files\PTQ-0803\results\test_fqvit_x4_bit3',
    r'D:\vscode-files\PTQ-0803\results\test_fqvit_x4_bit2',

]
res_file_fqvit = r'D:\vscode-files\PTQ-0803\stat_res\fqvit_res_bs32.txt'

index = 0

file_dirs = [
    file_dirs_minmaxquant,
    file_dirs_RepQ,
    file_dirs_fqvit,
][index]

res_file = [
    res_file_minmaxquant,
    res_file_RepQ,
    res_file_fqvit,
][index]

f_write = open(res_file, 'w')

cnt = 0 
for file_dir in file_dirs:
    if not os.path.exists(file_dir):
        continue
    for name in os.listdir(file_dir):
        if name.endswith('.log'):
            file_name = name
            break
    log_file_dir = os.path.join(file_dir, file_name)
    
    with open(log_file_dir, 'r') as log_file:
        text = log_file.read()
        pattern = r"# psnr: (\d+\.\d+)"
        psnrs = re.findall(pattern, text)
        
        pattern = r"# ssim: (\d+\.\d+)"
        ssims = re.findall(pattern, text)
        print(file_dir.split('\\')[-1])
        # print()
        for p,s in zip(psnrs, ssims):
            f_write.write(f'{float(p):.4f}/{float(s):.4f}\t')
            print(f'{float(p):.4f}/{float(s):.4f}\t', end='')
        print()         
        f_write.write('\n')
print(res_file)