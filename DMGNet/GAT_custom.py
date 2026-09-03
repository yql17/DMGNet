# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm
from torch_geometric.nn import global_mean_pool, BatchNorm
from models.Modified_GAT import GATConv as GATConv
from torch_geometric.nn import GraphSizeNorm

from models.model_utils import weight_init
from models.model_utils import decide_loss_type

from models.pre_layer import preprocess
from models.post_layer import postprocess

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
def drop_path(x, drop_prob: float = 0., training: bool = False):

    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output
class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
class TransformerBlock(nn.Module):
    """Transformer基础块：自注意力 + MLP"""
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        # 自注意力残差连接
        x = x + self.drop_path(self.attn(self.norm1(x)))
        # MLP残差连接
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


import torch
import torch.nn as nn
import torch.nn.functional as F

class GAT_module(torch.nn.Module):

    def __init__(self, input_dim, output_dim, head_num, dropedge_rate, graph_dropout_rate, loss_type, with_edge, simple_distance, norm_type):
        """
        :param input_dim: Input dimension for GAT
        :param output_dim: Output dimension for GAT
        :param head_num: number of heads for GAT
        :param dropedge_rate: Attention-level dropout rate
        :param graph_dropout_rate: Node/Edge feature drop rate
        :param loss_type: Choose the loss type
        :param with_edge: Include the edge feature or not
        :param simple_distance: Simple multiplication of edge feature or not
        :param norm_type: Normalization method
        """

        super(GAT_module, self).__init__()
        self.conv = GATConv([input_dim, input_dim], output_dim, heads=head_num, dropout=dropedge_rate, with_edge=with_edge, simple_distance=simple_distance)
        self.norm_type = norm_type
        if norm_type == "layer":
            self.bn = LayerNorm(output_dim * int(self.conv.heads))
            self.gbn = None
        else:
            self.bn = BatchNorm(output_dim * int(self.conv.heads))
            self.gbn = GraphSizeNorm()
        self.prelu = decide_loss_type(loss_type, output_dim * int(self.conv.heads))
        self.dropout_rate = graph_dropout_rate
        self.with_edge = with_edge

    def reset_parameters(self):

        self.conv.reset_parameters()
        self.bn.reset_parameters()

    def forward(self, x, edge_attr, edge_index, batch):

        if self.training:
            drop_node_mask = x.new_full((x.size(1),), 1 - self.dropout_rate, dtype=torch.float)
            drop_node_mask = torch.bernoulli(drop_node_mask)
            drop_node_mask = torch.reshape(drop_node_mask, (1, drop_node_mask.shape[0]))
            drop_node_feature = x * drop_node_mask

            drop_edge_mask = edge_attr.new_full((edge_attr.size(1),), 1 - self.dropout_rate, dtype=torch.float)
            drop_edge_mask = torch.bernoulli(drop_edge_mask)
            drop_edge_mask = torch.reshape(drop_edge_mask, (1, drop_edge_mask.shape[0]))
            drop_edge_attr = edge_attr * drop_edge_mask
        else:
            drop_node_feature = x
            drop_edge_attr = edge_attr

        if self.with_edge == "Y":
            x_before, attention_value = self.conv((drop_node_feature, drop_node_feature), edge_index,
                                   edge_attr=drop_edge_attr, return_attention_weights=True)
        else:
            x_before, attention_value = self.conv((drop_node_feature, drop_node_feature), edge_index,
                                   edge_attr=None, return_attention_weights=True)
        out_x_temp = 0
        if self.norm_type == "layer":
            for c, item in enumerate(torch.unique(batch)):
                temp = self.bn(x_before[batch == item])
                if c == 0:
                    out_x_temp = temp
                else:
                    out_x_temp = torch.cat((out_x_temp, temp), 0)
        else:
            temp = self.gbn(self.bn(x_before), batch)
            out_x_temp = temp

        x_after = self.prelu(out_x_temp)

        return x_after, attention_value

class GAT(torch.nn.Module):

    def __init__(self, dropout_rate, dropedge_rate, Argument):
        super(GAT, self).__init__()
        torch.manual_seed(12345)
        self.Argument = Argument

        dim = Argument.initial_dim
        self.dropout_rate = dropout_rate
        self.dropedge_rate = dropedge_rate
        self.heads_num = Argument.attention_head_num
        self.include_edge_feature = Argument.with_distance
        self.layer_num = Argument.number_of_layers
        self.graph_dropout_rate = Argument.graph_dropout_rate
        self.residual = Argument.residual_connection
        self.norm_type = Argument.norm_type
        self.attn_dim = dim * self.heads_num * (self.layer_num + 1)  # 计算x_concat的维度800维
        ###############################################################################
        self.topk_selector = TopKNodeSelector(k=Argument.topk_num)

        # 2. Transformer分支：处理原始输入特征（新增）
        # 原始特征维度 -> 投影到与GAT输出匹配的维度
        self.raw_feat_proj = nn.Linear(200, self.attn_dim)  # 原始特征投影

        # Transformer配置（2层，与GAT层数匹配）
        self.transformer = nn.Sequential(
            TransformerBlock(
                dim=self.attn_dim,
                num_heads=Argument.attn_heads,
                mlp_ratio=4.,
                qkv_bias=True,
                drop=dropout_rate,
                attn_drop=dropout_rate
            ),
            # TransformerBlock(
            #     dim=self.attn_dim,
            #     num_heads=Argument.attn_heads,
            #     mlp_ratio=4.,
            #     qkv_bias=True,
            #     drop=dropout_rate,
            #     attn_drop=dropout_rate
            # )
        )
        # Transformer输出归一化
        self.transformer_norm = LayerNorm(self.attn_dim)
        postNum = 0
        self.preprocess = preprocess(Argument)
        self.conv_list = nn.ModuleList([GAT_module(dim * self.heads_num, dim, self.heads_num, self.dropedge_rate,
                                                   self.graph_dropout_rate, Argument.loss_type,
                                                   with_edge=Argument.with_distance,
                                                   simple_distance=Argument.simple_distance,
                                                   norm_type=Argument.norm_type) for _ in
                                        range(int(Argument.number_of_layers))])
        postNum += int(self.heads_num) * len(self.conv_list)
        #----------- 新增：自注意力模块（输入维度为x_concat的维度）
        # x_concat的维度 = 初始全局特征维度 + 各层GAT全局特征维度之和

        self.self_attention = Attention(
            dim=self.attn_dim,  # 与x_concat维度匹配
            num_heads=Argument.attn_heads,  # 自注意力头数（可在Argument中配置）
            attn_drop=dropout_rate,
            proj_drop=dropout_rate
        )
        #-----------------
#MLP+风险预测层
        self.postprocess = postprocess(dim * self.heads_num, self.layer_num, dim * self.heads_num, (Argument.MLP_layernum-1), dropout_rate)

        self.risk_prediction_layer = nn.Linear(self.postprocess.postlayernum[-1], 1)

    # 在 GAT 类的 __init__ 中
        #self.alpha = nn.Parameter(torch.tensor(0.5))  # 可学习权重，初始化为0.5

    def reset_parameters(self):

        self.preprocess.reset_parameters()
        self.raw_feat_proj.reset_parameters()  # 新增：重置Transformer投影层
        for block in self.transformer:  # 新增：重置Transformer块
            block.norm1.reset_parameters()
            block.attn.reset_parameters()
            block.norm2.reset_parameters()
            block.mlp.fc1.reset_parameters()
            block.mlp.fc2.reset_parameters()
        self.transformer_norm.reset_parameters()  # 新增：重置Transformer归一化
        for i in range(int(self.Argument.number_of_layers)):
            self.conv_list[i].reset_parameters()
        self.self_attention.reset_parameters()  # 重置自注意力参数
        self.postprocess.reset_parameters()
        self.risk_prediction_layer.apply(weight_init)

    def forward(self, data, edge_mask=None, Interpretation_mode=False):

        row, col, _ = data.adj_t.coo()
#预处理，降维到（9507，200）边特征维度为200
        preprocessed_input, preprocess_edge_attr = self.preprocess(data, edge_mask)
        batch = data.batch
        B = batch.max().item() + 1  # 批次大小
        #######################################################################
        # Transformer分支：先筛选Top-K节点，再送入Transformer
        #######################################################################
        # 1. 原始特征投影：(Total_Nodes, 200) -> (Total_Nodes, 800)
        raw_feat = self.raw_feat_proj(preprocessed_input)

      
        topk_nodes = self.topk_selector(raw_feat, batch) 
        # 3. Transformer处理筛选后的节点特征
        trans_feat = self.transformer(topk_nodes)  # (B, K, 800)
        trans_feat = self.transformer_norm(trans_feat)  # 归一化
        # 4. 全局池化：(B, K, 800) -> (B, 800)（按节点维度平均）
        trans_glob = trans_feat.mean(dim=1)  # 新增：对筛选后的节点平均池化
        ###############################################################################
###########################################################################
#全局池化（N,200）
        x0_glob = global_mean_pool(preprocessed_input, batch)
        x_concat = x0_glob

        x_out = preprocessed_input
        final_x = x_out
        count = 0
        attention_list = []

        for i in range(int(self.layer_num)):
            select_idx = int(i)
            x_temp_out, attention_value = \
                self.conv_list[select_idx](x_out, preprocess_edge_attr, data.adj_t, batch)
            _, _, attention_value = attention_value.coo()
            if len(attention_list) == 0:
                attention_list = torch.reshape(attention_value, (1, attention_value.shape[0], attention_value.shape[1]))
            else:
                attention_list = torch.cat((attention_list, torch.reshape(attention_value, (
                1, attention_value.shape[0], attention_value.shape[1]))), 0)

            x_glob = global_mean_pool(x_temp_out, batch)
            x_concat = torch.cat((x_concat, x_glob), 1)

            if self.residual == "Y":
                x_out = x_temp_out + x_out
            else:
                x_out = x_temp_out

            final_x = x_out #最终一层图特征
            count = count + 1

        #print(f"x_concat shape: {x_concat.shape}")
        x_concat_reshaped = x_concat.unsqueeze(1)  # 重塑为 (B, 1, C)
       # attn_out = self.self_attention(x_concat_reshaped)  # 自注意力输出 (B, 1, C)
       # attn_out = attn_out.squeeze(1)  # 压缩为 (B, C)，与原x_concat形状一致

        # 融合阶段
        fused_glob = self.alpha * x_concat + (1 - self.alpha) * trans_glob

        #print(f"attn_out shape: {attn_out.shape}")
        #postprocessed_output = self.postprocess(x_concat, data.batch)
        # 替换x_concat为attn_out
        postprocessed_output = self.postprocess(trans_glob, data.batch)
        risk = self.risk_prediction_layer(postprocessed_output)

        if Interpretation_mode:
            return risk, final_x, attention_list
        else:
            return risk,fused_glob