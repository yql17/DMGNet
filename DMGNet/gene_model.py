# -*- coding:utf8 -*-
import random
import warnings

from sklearn.preprocessing import StandardScaler

from Superpatch_network_construction.utils.util import CoxLoss


warnings.filterwarnings("ignore")
import sys
import os
# 获取项目根路径（根据实际目录结构调整，这里假设 project02 所在的 project 目录的上级是根路径）
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(root_path)
# 然后用绝对导入

from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sksurv.metrics import concordance_index_censored
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from scipy.stats import spearmanr

def set_seed(seed=12345):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # for single-GPU
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # disable to make deterministic
    os.environ['PYTHONHASHSEED'] = str(seed)


class SNN_GAT(nn.Module):
    def __init__(self, nfeat, nhid1, output, num_class, dropout=0.5):
        super(SNN_GAT, self).__init__()
        #torch.manual_seed(12345)
        self.GAT = GAT(nfeat, nhid1, output, dropout)
        self.MLP = MLP(output, num_class, dropout)

    def forward(self, features, adj):
        device = next(self.parameters()).device
        features = torch.tensor(features, dtype=torch.float32, device=device)
        adj = adj.to(device)
        indices = torch.nonzero(adj, as_tuple=False).t()
        edge_index = indices[0:2].to(torch.long)
        data = Data(x=features, edge_index=edge_index)
        gat_output = self.GAT(data)
        Label = self.MLP(gat_output)
        return Label, gat_output


def generate_graph_data(gene_df, k=35, use_cosine=True, mutual=True, verbose=True):


    # 1. 相似度计算
    if use_cosine:
        sim_matrix = cosine_similarity(gene_df)
    else:
        dist_matrix = euclidean_distances(gene_df)
        sigma = np.median(dist_matrix)
        sim_matrix = np.exp(-dist_matrix ** 2 / (2 * sigma ** 2))

    n = sim_matrix.shape[0]
    adj_matrix = np.zeros((n, n), dtype=np.float32)

    for i in range(n):
        # 当前样本前k个最大相似度的索引（排除自己）
        knn_indices = np.argsort(sim_matrix[i])[::-1][1:k+1]
        adj_matrix[i, knn_indices] = sim_matrix[i, knn_indices]

    if mutual:
        # 互选策略：只有 i->j 且 j->i 都是邻居才保留边
        mutual_adj = np.minimum(adj_matrix, adj_matrix.T)
        adj_matrix = mutual_adj

    # 保留最大值为1
    adj_matrix /= np.max(adj_matrix)

    if verbose:
        sparsity = 1.0 - np.count_nonzero(adj_matrix) / (n * n)
        degree = np.sum(adj_matrix > 0, axis=1)
        print(f"[SNN] 稀疏度: {sparsity:.4f} | 平均度: {np.mean(degree):.2f} | 最大度: {np.max(degree)}, 最小度: {np.min(degree)}")

    # 转为 PyTorch 张量
    node_features = torch.FloatTensor(gene_df.values.astype(np.float32))
    adj_tensor = torch.FloatTensor(adj_matrix)

    return node_features, adj_tensor

def analyze_snn_structure(adj_matrix):
    if isinstance(adj_matrix, torch.Tensor):
        adj_matrix = adj_matrix.cpu().numpy()
    sparsity = 1.0 - np.count_nonzero(adj_matrix) / adj_matrix.size
    print(f"[SNN] 稀疏度: {sparsity:.4f}")
    degrees = np.sum(adj_matrix > 0, axis=1)
    print(f"[SNN] 平均度: {np.mean(degrees):.2f}，最大度: {np.max(degrees)}, 最小度: {np.min(degrees)}")
    plt.figure(figsize=(6, 5))
    sns.heatmap(adj_matrix[:100, :100], cmap='viridis')
    plt.title("SNN 前100个样本的相似度矩阵")
    plt.tight_layout()
    plt.show()

def visualize_embeddings(original_features, embedded_features, labels, title=""):
    original_emb = TSNE(n_components=2, random_state=42).fit_transform(original_features.cpu().numpy())
    embedded_emb = TSNE(n_components=2, random_state=42).fit_transform(embedded_features.cpu().detach().numpy())
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, emb, name in zip(axes, [original_emb, embedded_emb], ["Original Space", "GAT"]):
        sc = ax.scatter(emb[:, 0], emb[:, 1], c=labels.cpu().numpy(), cmap='coolwarm', s=10)
        ax.set_title(name)
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()

def evaluate_feature_label_corr(embedding, survival_time):
    embedding_np = embedding.cpu().detach().numpy()
    survival_np = survival_time.cpu().numpy()
    corrs = [spearmanr(embedding_np[:, i], survival_np)[0] for i in range(embedding_np.shape[1])]
    print(f"[SNN] 平均特征-生存时间相关系数: {np.mean(np.abs(corrs)):.4f}")

class EarlyStopping:
    def __init__(self, patience=20, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model_wts = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model_wts = model.state_dict()
        elif val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.best_model_wts = model.state_dict()
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


    