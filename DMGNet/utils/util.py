# -*- coding:utf8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init


def make_batch(batch_graph):
    batch_masks = []
    nodes = batch_graph['nodes']
    bs = len(nodes)
    batch_edges = batch_graph['edges']
    #batch_slide_names = batch_graph['slide_name']
    batch_slide_names = batch_graph['id']
    #batch_OS_times = torch.tensor(batch_graph['OS_time'])
    batch_OS_times = torch.tensor(batch_graph['os'])
    #batch_OSs = torch.tensor(batch_graph['OS'])
    batch_OSs = torch.tensor(batch_graph['states'])
    max_nodes_num = 0
    for i in range(bs):
        max_nodes_num = max(max_nodes_num, nodes[i].shape[0])
        batch_masks.append(nodes[i].shape[0])#batch_masks 是一个长度为 batch_size 的列表，用于记录每个 WSI 图中有多少个 patch（节点）
    batch_masks = torch.tensor(batch_masks)
    #生成统一形状的张量 batch_nodes，形状为 [batch_size, max_patch_num, feat_dim]
    batch_nodes = torch.zeros(bs, max_nodes_num, nodes[0].shape[1])
    for i in range(bs):
        num = batch_masks[i]
        batch_nodes[i][0:num] = nodes[i]

    return {'batch_nodes': batch_nodes, 'batch_edges': batch_edges,
            'batch_masks': batch_masks}, batch_OSs, batch_OS_times, batch_slide_names

def make_batch_(batch_graph):
    batch_masks = []
    nodes = batch_graph['nodes']
    bs = len(nodes)
    batch_edges = batch_graph['edges']
    # batch_slide_names = batch_graph['slide_name']
    # batch_OS_times = torch.tensor(batch_graph['OS_time'])
    # batch_OSs = torch.tensor(batch_graph['OS'])
    batch_slide_names = batch_graph['id']
    batch_OS_times = torch.tensor(batch_graph['os'])
    batch_OSs = torch.tensor(batch_graph['states'])
    max_nodes_num = 0
    for i in range(bs):
        max_nodes_num = max(max_nodes_num, nodes[i].shape[0])
        batch_masks.append(nodes[i].shape[0])
    batch_masks = torch.tensor(batch_masks)
    batch_nodes = torch.zeros(bs, max_nodes_num, nodes[0].shape[1])
    for i in range(bs):
        num = batch_masks[i]
        batch_nodes[i][0:num] = nodes[i]

    return (batch_nodes, batch_edges, batch_masks), batch_OSs, batch_OS_times, batch_slide_names


def collate(batch):
    nodes = [b['node'] for b in batch]
    edges = [b['edge'] for b in batch]
    # states = [b['state'] for b in batch]
    # os = [b['os'] for b in batch]
    # id = [b['id'] for b in batch]
    states = [b['OS'] for b in batch]
    os = [b['OS_time'] for b in batch]
    id = [b['slide_name'] for b in batch]
    return {'nodes': nodes, 'edges': edges, 'states': states, 'os': os, 'id': id}


def CoxLoss(survtime, censor, hazard_pred):
    # This calculation credit to Travers Ching https://github.com/traversc/cox-nnet
    # Cox-nnet: An artificial neural network method for prognosis prediction of high-throughput omics data
    current_batch_len = len(survtime)#当前批次中的样本数量
    R_mat = np.zeros([current_batch_len, current_batch_len], dtype=int)#初始化风险集矩阵，R_mat 是一个二维的零矩阵
    #对于 R_mat[i, j]，若 survtime[j] 大于或等于 survtime[i]，则将其赋值为 1，反之赋值为 0。此矩阵用于确定每个样本的风险集。
    for i in range(current_batch_len):
        for j in range(current_batch_len):
            R_mat[i, j] = survtime[j] >= survtime[i]
    device = survtime.device
    #R_mat = torch.FloatTensor(R_mat).cuda()
    R_mat = torch.FloatTensor(R_mat).to(device)
    theta = hazard_pred.reshape(-1)

    exp_theta = torch.exp(theta)
    loss_cox = -torch.mean((theta - torch.log(torch.sum(exp_theta * R_mat, dim=1))) * censor)
    return loss_cox


def weight_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
