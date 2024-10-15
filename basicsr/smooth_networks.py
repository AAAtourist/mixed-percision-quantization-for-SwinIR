import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import KMeans
import wandb

wandb.init(project='smooth SwinIR')


config = wandb.config
config.learning_rate = 0.01
config.epochs = 100

def compute_channel_statistics(tensors):
    means = torch.mean(tensors, dim=1)  #[2048, 60]
    stds = torch.std(tensors, dim=1)    #[2048, 60]
    return torch.cat([means, stds], dim=-1)  #[2048, 120]

def initialize_matrices(size, noise_scale=0.01):
    A = nn.Parameter(torch.eye(size) + noise_scale * torch.randn(size, size))
    B = nn.Parameter(torch.eye(size) + noise_scale * torch.randn(size, size))
    return A, B

def orthogonality_loss(A, B):
    I = torch.eye(A.size(0), device=A.device)
    return torch.norm(torch.matmul(A, B) - I, p='fro') ** 2 #forb

def smoothness_loss_XA(XA):
    return ((XA[:, :, :-1] - XA[:, :, 1:])**2).mean() + ((XA[:, :-1, :] - XA[:, 1:, :])**2).mean()

def smoothness_loss_BW(BW):
    return ((BW[:, :-1] - BW[:, 1:])**2).mean() + ((BW[:-1, :] - BW[1:, :])**2).mean()

def range_consistency_loss(group_outputs):
    ranges = [torch.max(output) - torch.min(output) for output in group_outputs]
    return torch.var(torch.stack(ranges))

def apply_group_transform(X_group, weight, A, B):
    XA = torch.matmul(X_group, A)
    BW = torch.matmul(B, weight)
    return XA, BW

def total_loss(grouped_matrices, group_labels, weight, A_matrices, B_matrices, weight_smooth=0.1, weight_range=0.1, weight_ortho=0.1):
    group_outputs_XA = []
    group_outputs_BW = []
    smooth_losses_XA = []
    smooth_losses_BW = []
    ortho_losses = []
    
    for i in range(len(A_matrices)):
        group_indices = torch.where(group_labels == i)[0]
        X_group = grouped_matrices[group_indices]
        A = A_matrices[i]
        B = B_matrices[i]
        
        XA, BW = apply_group_transform(X_group, weight, A, B)
        group_outputs_XA.append(XA)
        group_outputs_BW.append(BW)
        
        smooth_losses_XA.append(smoothness_loss_XA(XA))
        smooth_losses_BW.append(smoothness_loss_BW(BW))
        
        ortho_losses.append(orthogonality_loss(A, B))
    
    range_loss_XA = range_consistency_loss(group_outputs_XA)
    range_loss_BW = range_consistency_loss(group_outputs_BW)
    
    total_loss = (sum(smooth_losses_XA) + sum(smooth_losses_BW)) * weight_smooth \
                 + (range_loss_XA + range_loss_BW) * weight_range \
                 + sum(ortho_losses) * weight_ortho
                 
    return total_loss, smooth_losses_XA, smooth_losses_BW, range_loss_XA, range_loss_BW, ortho_losses

def train(X, weight, group_labels, weight_smooth=0.1, weight_range=0.1, weight_ortho=0.1, learning_rate=0.001):
    group_count = len(set(group_labels.cpu().numpy()))
    
    A_matrices = []
    B_matrices = []
    for _ in range(group_count):
        A, B = initialize_matrices(60)
        A_matrices.append(A)
        B_matrices.append(B)
    
    optimizer = torch.optim.Adam(A_matrices + B_matrices, lr=learning_rate)
    
    for epoch in range(config.epochs):
        optimizer.zero_grad()

        loss, smooth_losses_XA, smooth_losses_BW, range_loss_XA, range_loss_BW, ortho_losses = total_loss(
            X, group_labels, weight, A_matrices, B_matrices, weight_smooth, weight_range, weight_ortho
        )

        loss.backward()
        optimizer.step()
        
        wandb.log({'epoch': epoch, 'Total Loss': loss.item(), 'Range Loss XA': range_loss_XA.item(), 'Range Loss BW' :range_loss_BW.item(), 'Ortho Loss': sum(ortho_losses).item()})

    return A_matrices, B_matrices

def assign_groups(statistics, kmeans):
    return torch.tensor(kmeans.predict(statistics.cpu().numpy()))

class smooth_network(nn.Module):
    def __init__(self, W, clusters):
        super(smooth_network, self).__init__()

        self.input = None
        self.weight = W
        self.clusters = clusters
        self.kmeans = None
        self.A_matrices = None
        self.B_matrices = None



    def forward(self, X):

        XA_result, BW_result = self.process_activation(X)
        
        return XA_result, BW_result

    def inited(self, X):
        statistics = compute_channel_statistics(X)

        self.kmeans = KMeans(n_clusters=self.clusters, random_state=0)
        group_labels = torch.tensor(self.kmeans.fit_predict(statistics.cpu().numpy()))
        #print(group_labels)

        self.A_matrices, self.B_matrices = train(X, self.weight, group_labels, weight_smooth=0.1, 
                                                weight_range=0.1, weight_ortho=0.1, learning_rate=0.001)

    def process_activation(self, X):
        statistics = compute_channel_statistics(X)
    
        group_labels = assign_groups(statistics, self.kmeans) 

        XA_list = []
        BW_list = []

        for i in range(len(X)):
            group_id = group_labels[i].item()
            X_group = X[i]
            A = self.A_matrices[group_id]
            B = self.B_matrices[group_id]
            
            XA, BW = apply_group_transform(X_group, self.weight, A, B)
            
            XA_list.append(XA)
            BW_list.append(BW)
        
        XA_result = torch.stack(XA_list)
        BW_result = torch.stack(BW_list)
        
        return XA_result, BW_result
    
