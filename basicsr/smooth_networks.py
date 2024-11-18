import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import KMeans, MiniBatchKMeans
import wandb
import numpy as np

# 加实验名 
#wandb.init(project='smooth SwinIR')


#config = wandb.config
#config.learning_rate = 0.01
#config.epochs = 300

def lp_loss(pred, tgt):
    res = float(torch.norm(pred - tgt) / (pred - tgt).numel())
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
    #means = means / torch.quantile(means, 0.6).detach()
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
    return torch.norm(torch.matmul(A, B) - I, p=2) / A.size(0)

'''def range_consistency_loss(group_outputs):
    ranges = [torch.max(output) - torch.min(output) for output in group_outputs]
    return torch.var(torch.stack(ranges))'''

def l2_smooth_loss_full_3d(x):
    diff_b = x[1:, :, :] - x[:-1, :, :]
    diff_c = x[:, 1:, :] - x[:, :-1, :]
    diff_t = x[:, :, 1:] - x[:, :, :-1]

    loss = torch.mean(diff_b ** 2) + torch.mean(diff_c ** 2) + torch.mean(diff_t ** 2)
    return loss

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


class smooth_network(nn.Module):
    _instance = None 

    def __new__(cls, clusters=None, emb_dim=None, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(smooth_network, cls).__new__(cls)
        return cls._instance

    def __init__(self, clusters=None, emb_dim=None):
        if not hasattr(self, 'initialized'): 
            super(smooth_network, self).__init__()

            if clusters is None or emb_dim is None:
                raise ValueError("clusters and emb_dim must be provided during the first initialization.")

            self.clusters = clusters
            self.kmeans = MiniBatchKMeans(n_clusters=self.clusters, random_state=0)
            self.A_matrices = nn.ParameterList()
            self.B_matrices = nn.ParameterList()

            for _ in range(self.clusters):
                A, B = initialize_matrices(emb_dim)
                A = nn.Parameter(A.cuda(), requires_grad=True)
                B = nn.Parameter(B.cuda(), requires_grad=True)
                self.A_matrices.append(A)
                self.B_matrices.append(B)

            self.initialized = True  

    def forward(self, X, weight):
        assert not torch.isnan(X).any(), 'nan X'
        statistics = compute_channel_statistics(X)  # [batc[h_size, 120]
        group_labels = self.assign_groups(statistics)  # [batch_size]
        #print(group_labels)

        batch_size = X.size(0)
        
        XA_result = torch.zeros_like(X)  # [batch_size, 64, 60]
        BW_result = torch.zeros((batch_size, *weight.shape), device=X.device)  # [batch_size, 60, 60]

        #print(group_labels)
        #A_expanded = self.A_matrices[group_labels]  # [batch_size, 60, 60]
        #B_expanded = self.B_matrices[group_labels]  # [batch_size, 60, 60]
        A_matrices_tensor = torch.stack([param for param in self.A_matrices])
        B_matrices_tensor = torch.stack([param for param in self.B_matrices])

        A_expanded = A_matrices_tensor[group_labels]
        B_expanded = B_matrices_tensor[group_labels]

        XA_result = torch.bmm(X, A_expanded)  # [batch_size, 64, 60]
        BW_result = torch.bmm(B_expanded, weight.unsqueeze(0).expand(X.size(0), -1, -1))

        '''
            for group_id in range(len(self.A_matrices)):
            mask = (group_labels == group_id)
            
            if mask.sum() == 0:
                continue
            
            X_group = X[mask]  # [num_in_group, 64, 60]
            
            A = self.A_matrices[group_id]  # [60, 60]
            B = self.B_matrices[group_id]  # [60, 60]
            
            XA_result[mask] = torch.bmm(X_group, A.unsqueeze(0).expand(X_group.size(0), -1, -1))
            BW_result[mask] = torch.bmm(B.unsqueeze(0).expand(X_group.size(0), -1, -1), self.weight.unsqueeze(0).expand(X_group.size(0), -1, -1))'''
        
        #result = torch.bmm(input_quantizer(XA_result), weight_qunantizer(BW_result))
        return XA_result, BW_result
    
    def assign_groups(self, statistics:torch.Tensor):
        return torch.tensor(self.kmeans.predict(statistics.cpu().detach().numpy()))

    def inited(self, X):
        
        X = X.reshape(-1, 64, X.shape[-1]) #64 window_size
        statistics = compute_channel_statistics(X).cpu().numpy()
        #statistics_np = statistics.cpu().numpy()

        #self.kmeans = KMeans(n_clusters=self.clusters, random_state=0)
        self.kmeans.batch_size = statistics.shape[0]

        self.kmeans.partial_fit(statistics)

    def orth_loss(self):
       # A_matrices_tensor = torch.stack([param.data for param in self.A_matrices])
        #B_matrices_tensor = torch.stack([param.data for param in self.B_matrices])

        loss = 0
        for i in range(self.clusters):
            loss += orthogonality_loss(self.A_matrices[i], self.B_matrices[i]) / self.clusters

        return loss

    '''
        def total_loss(self, X, group_labels, input_quantizer, weight_qunantizer):
            group_outputs_XA = []
            group_outputs_BW = []
            quant_losses = []
            ortho_losses = []

            XA_result = torch.zeros_like(X)  # [batch_size, 64, 60]
            BW_result = torch.zeros((X.size(0), *self.weight.shape), device=X.device)  # [batch_size, 60, 60]

            A_expanded = self.A_matrices[group_labels]  # [batch_size, 60, 60]
            B_expanded = self.B_matrices[group_labels]  # [batch_size, 60, 60]

            XA_result = torch.bmm(X, A_expanded)  # [batch_size, 64, 60]
            BW_result = torch.bmm(B_expanded, self.weight.unsqueeze(0).expand(X.size(0), -1, -1))  # [batch_size, 60, 60]

            
            smooth_losses_XA = torch.var(XA_result)
            smooth_losses_BW = torch.var(BW_result)

                for group_id in range(len(self.A_matrices)):
                mask = (group_labels == group_id)
                if mask.sum() == 0:
                    continue
                
                X_group = X[mask]  # [num_in_group, 64, 60]
                
                A = self.A_matrices[group_id]  # [60, 60]
                B = self.B_matrices[group_id]  # [60, 60]

                ortho_losses.append(orthogonality_loss(A, B))
                
                XA_result[mask] = torch.bmm(X_group, A.unsqueeze(0).expand(X_group.size(0), -1, -1))
                BW_result[mask] = torch.bmm(B.unsqueeze(0).expand(X_group.size(0), -1, -1), self.weight.unsqueeze(0).expand(X_group.size(0), -1, -1))
            
                #result = torch.bmm(input_quantizer(XA_result), weight_qunantizer(BW_result))
                #smooth_losses_XA = l2_smooth_loss_full_3d(XA_result)
                #smooth_losses_BW = l2_smooth_loss_full_3d(BW_result)
                smooth_losses_XA = torch.var(XA_result)
                smooth_losses_BW = torch.var(BW_result)
                #quant_losses = lp_loss(result, X @ self.weight)
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
            #smooth_losses_XA = torch.tensor(smooth_losses_XA, dtype=torch.float32)
            #smooth_losses_BW = torch.tensor(smooth_losses_BW, dtype=torch.float32)
            #quant_losses = torch.tensor(quant_losses, dtype=torch.float32)
            ortho_losses = torch.tensor(ortho_losses, dtype=torch.float32)
            
            total_loss = (sum(smooth_losses_XA)*2 + sum(smooth_losses_BW)* 6) * 0.5 \
                        + (range_loss_XA * 3 + range_loss_BW * 10) * 0.1 \
                        + sum(ortho_losses) * 0.1
            #print('smooth_loss', torch.mean(smooth_losses_XA) + torch.mean(smooth_losses_BW))
            #print('range_loss', (range_loss_XA  + range_loss_BW ))
            print('orth_loss', torch.mean(ortho_losses))

            #print('smoothloss', smooth_losses_XA + smooth_losses_BW)
            #print('quant_loss', quant_losses)
            #print('orth_loss', torch.mean(ortho_losses))
            total_loss = smooth_losses_XA + smooth_losses_BW \
                        #+ torch.mean(ortho_losses) \
                        #+ quant_losses
                        #+ (range_loss_XA  + range_loss_BW ) \

            return total_loss
            '''