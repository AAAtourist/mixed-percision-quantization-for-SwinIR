import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import KMeans
import wandb
import numpy as np

# 加实验名 
#wandb.init(project='smooth SwinIR')


#config = wandb.config
#config.learning_rate = 0.01
#config.epochs = 300

def lp_loss(pred, tgt):
    res = float(torch.norm(pred - tgt))
    return res


'''def _quantize(x):
    n_bits = 2
    x_clone = x.clone().detach()
    x_max = x_clone.max()
    x_min = x_clone.min()
    best_score = 1e+10
    best_delta = None
    best_zp = None
    delta = (x_max - x_min) / (2 ** n_bits - 1)
    zero_point = (- x_min / delta).round()
    for pct in [0.9, 0.99, 0.999, 0.9999, 0.99999]:
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

        x_q = quantize(x_clone, new_max, new_min)
        score = lp_loss(x_clone, x_q, p=2)

        if score < best_score:
            best_score = score
            best_delta = (new_max - new_min) / (2 ** n_bits - 1)
            best_zp = (- new_min / delta).round()

    x_int = torch.round(x / best_delta) + best_zp
    x_quant = torch.clamp(x_int, 0, 2 ** n_bits - 1)
    x_dequant = (x_quant - best_zp) * best_delta

    return x_dequant'''


def compute_channel_statistics(tensors):
    means = torch.mean(tensors, dim=1)  #[2048, 60]
    stds = torch.std(tensors, dim=1)    #[2048, 60]
    return torch.cat([means, stds], dim=-1)  #[2048, 120]

def initialize_matrices(size, noise_scale=0.01):
    
    #A = nn.Parameter(torch.eye(size) + noise_scale * torch.randn(size, size), requires_grad=True)
    #B = nn.Parameter(torch.eye(size) + noise_scale * torch.randn(size, size), requires_grad=True)

    A = nn.Parameter(torch.eye(size))
    B = nn.Parameter(torch.eye(size))
    return A, B

def orthogonality_loss(A, B):
    I = torch.eye(A.size(0), device=A.device)
    return torch.norm(torch.matmul(A, B) - I, p='fro') ** 2 #forb

'''def range_consistency_loss(group_outputs):
    ranges = [torch.max(output) - torch.min(output) for output in group_outputs]
    return torch.var(torch.stack(ranges))'''

def range_consistency_loss(group_outputs):
    ranges1 = [torch.max(output) for output in group_outputs]
    ranges2 = [torch.min(output) for output in group_outputs]
    #print(f"stacked_ranges1 shape: {len(ranges1)}, stacked_ranges2 shape: {len(ranges2)}")
    return torch.var(torch.stack(ranges1)) + torch.var(torch.stack(ranges2))

'''def total_loss(grouped_matrices, group_labels, weight, A_matrices, B_matrices, weight_smooth=0.1, weight_range=0.1, weight_ortho=0.1):
    group_outputs_XA = []
    group_outputs_BW = []
    smooth_losses_XA = []
    smooth_losses_BW = []
    quant_losses = []
    ortho_losses = []
    
    for i in range(len(A_matrices)):
        group_indices = torch.where(group_labels == i)[0]
        X_group = grouped_matrices[group_indices]
        A = A_matrices[i]
        B = B_matrices[i]
        
        XA = torch.matmul(X_group, A)
        BW = torch.matmul(B, weight)

        origin = torch.matmul(X_group, weight)

        group_outputs_XA.append(XA)
        group_outputs_BW.append(BW)


        smooth_losses_XA.append(torch.var(XA))
        smooth_losses_BW.append(torch.var(BW))

        xa = quantize(XA)
        bw = quantize(BW)
        
        quant_losses.append(lp_loss(xa @ bw, origin))
        
        ortho_losses.append(orthogonality_loss(A, B))
    
    range_loss_XA = range_consistency_loss(group_outputs_XA)
    range_loss_BW = range_consistency_loss(group_outputs_BW)
    
    total_loss = (sum(smooth_losses_XA)*2 + sum(smooth_losses_BW)* 6) * weight_smooth \
                 + (range_loss_XA * 3 + range_loss_BW * 10) * weight_range \
                 + sum(quant_losses) * 100 \
                 + sum(ortho_losses) * weight_ortho
                 
    return total_loss, quant_losses, smooth_losses_XA, smooth_losses_BW, range_loss_XA, range_loss_BW, ortho_losses

def train(X, weight, group_labels, weight_smooth=0.5, weight_range=0.1, weight_ortho=0.1, learning_rate=0.0001):
    group_count = len(set(group_labels.cpu().numpy()))
    
    X = X.detach()
    X.requires_grad = False
    
    weight = weight.detach()
    weight.requires_grad = False

    A_matrices = []# 2048 64 60
    B_matrices = []
    for _ in range(group_count):
        A, B = initialize_matrices(X.shape[-1])
        A = nn.Parameter(A.cuda(), requires_grad=True)
        B = nn.Parameter(B.cuda(), requires_grad=True)
        A_matrices.append(A)
        B_matrices.append(B)
    
    optimizer = torch.optim.Adam(A_matrices + B_matrices, lr=learning_rate)

    
    for epoch in range(200):
        optimizer.zero_grad()

        loss, quant_loss, smooth_losses_XA, smooth_losses_BW, range_loss_XA, range_loss_BW, ortho_losses = total_loss(
            X, group_labels, weight, A_matrices, B_matrices, weight_smooth, weight_range, weight_ortho
        )

        loss.backward()
        optimizer.step()
        #wandb.log({'smooth_losses_XA_': sum(smooth_losses_XA).item(), 'smooth_losses_BW': sum(smooth_losses_BW), 'quant_loss': sum(quant_loss)})
        #wandb.log({'epoch': epoch, 'Total Loss': loss.item(), 'Range Loss XA': range_loss_XA.item(), 'Range Loss BW' :range_loss_BW.item(), 'Ortho Loss': sum(ortho_losses).item()})

    return A_matrices, B_matrices
'''
def assign_groups(statistics, kmeans):
    return torch.tensor(kmeans.predict(statistics.cpu().detach().numpy()))

class smooth_network(nn.Module):
    def __init__(self, W, clusters, bit):
        super(smooth_network, self).__init__()

        self.input = None
        self.weight = W
        self.clusters = clusters
        self.kmeans = None 
        self.A_matrices = nn.ParameterList()
        self.bit = bit
        self.B_matrices = nn.ParameterList() # 2048 64 60 x 



    def forward(self, X, input_quantizer, weight_qunantizer):
        #print(X.shape)
        
        
        assert not torch.isnan(X).any(), 'nan X'
        statistics = compute_channel_statistics(X)  # [batch_size, 120]
        group_labels = assign_groups(statistics, self.kmeans)  # [batch_size]

        batch_size = X.size(0)
        
        XA_result = torch.zeros_like(X)  # [batch_size, 64, 60]
        BW_result = torch.zeros((batch_size, *self.weight.shape), device=X.device)  # [batch_size, 60, 60]
        loss = None if not self.training else self.total_loss(X, group_labels, input_quantizer, weight_qunantizer)

        for group_id in range(len(self.A_matrices)):
            mask = (group_labels == group_id)
            
            if mask.sum() == 0:
                continue
            
            X_group = X[mask]  # [num_in_group, 64, 60]
            
            A = self.A_matrices[group_id]  # [60, 60]
            B = self.B_matrices[group_id]  # [60, 60]
            
            XA_result[mask] = torch.bmm(X_group, A.unsqueeze(0).expand(X_group.size(0), -1, -1))
            BW_result[mask] = torch.bmm(B.unsqueeze(0).expand(X_group.size(0), -1, -1), self.weight.unsqueeze(0).expand(X_group.size(0), -1, -1))
        
        result = torch.bmm(input_quantizer(XA_result), weight_qunantizer(BW_result))
        return result, loss

    def inited(self, X):

        X = X.reshape(-1, 64, X.shape[-1]) #64 window_size
        self.weight = self.weight.to(X.device)
        statistics = compute_channel_statistics(X) 

        self.kmeans = KMeans(n_clusters=self.clusters, random_state=0)

        for _ in range(self.clusters):
            A, B = initialize_matrices(X.shape[-1])
            A = nn.Parameter(A.cuda(), requires_grad=True)
            B = nn.Parameter(B.cuda(), requires_grad=True)
            self.A_matrices.append(A)
            self.B_matrices.append(B)

        group_labels = torch.tensor(self.kmeans.fit_predict(statistics.cpu().numpy()))

        #print(group_labels)

        #self.A_matrices, self.B_matrices = train(X, self.weight, group_labels, weight_smooth=10, 
        #                                        weight_range=20, weight_ortho=10, learning_rate=0.02)

    
    def total_loss(self, X, group_labels, input_quantizer, weight_qunantizer):
        group_outputs_XA = []
        group_outputs_BW = []
        smooth_losses_XA = []
        smooth_losses_BW = []
        quant_losses = []
        ortho_losses = []
        
        for i in range(len(self.A_matrices)):
            group_indices = torch.where(group_labels == i)[0]
            if not group_indices.numel():
                continue

            X_group = X[group_indices]

            A = self.A_matrices[i]
            B = self.B_matrices[i]
            
            XA = torch.matmul(X_group, A)
            BW = torch.matmul(B, self.weight)
            
            origin = torch.matmul(X_group, self.weight)

            #group_outputs_XA.append(XA)
            #group_outputs_BW.append(BW)

            smooth_losses_XA.append(torch.var(XA))
            smooth_losses_BW.append(torch.var(BW))

            xa = input_quantizer(XA)
            bw = weight_qunantizer(BW)
            
            quant_losses.append(lp_loss(xa @ bw, origin)) #norm
            
            ortho_losses.append(orthogonality_loss(A, B))
        
        #range_loss_XA = range_consistency_loss(group_outputs_XA)
        #range_loss_BW = range_consistency_loss(group_outputs_BW)

        #range_loss_XA = torch.tensor(range_loss_XA, dtype=torch.float32)
        #range_loss_BW = torch.tensor(range_loss_BW, dtype=torch.float32)
        smooth_losses_XA = torch.tensor(smooth_losses_XA, dtype=torch.float32)
        smooth_losses_BW = torch.tensor(smooth_losses_BW, dtype=torch.float32)
        quant_losses = torch.tensor(quant_losses, dtype=torch.float32)
        ortho_losses = torch.tensor(ortho_losses, dtype=torch.float32)
        
        '''total_loss = (sum(smooth_losses_XA)*2 + sum(smooth_losses_BW)* 6) * 0.5 \
                    + (range_loss_XA * 3 + range_loss_BW * 10) * 0.1 \
                    + sum(ortho_losses) * 0.1'''
        #print('smooth_loss', torch.mean(smooth_losses_XA) + torch.mean(smooth_losses_BW))
        #print('range_loss', (range_loss_XA  + range_loss_BW ))
        #print('orth_loss', torch.mean(ortho_losses))


        total_loss = (torch.mean(smooth_losses_XA) + torch.mean(smooth_losses_BW)) \
                    + torch.mean(ortho_losses) + quant_losses
                    #+ (range_loss_XA  + range_loss_BW ) \

        return total_loss