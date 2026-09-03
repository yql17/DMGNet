#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 21:10:37 2021

@author: kyungsub
"""

import os

import h5py
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import torch
import openslide as osd
from torchvision import transforms
from torch_geometric.data import Data
from EfficientNet import EfficientNet
from superpatch_network_construction import false_graph_filtering
from skimage.filters import threshold_multiotsu
import pickle
import argparse

class SurvivalImageDataset():

    """
    Target dataset has the list of images such as
    _patientID_SurvDay_Censor_TumorStage_WSIPos.tif
    """

    def __init__(self, image, x, y, transform):

        self.image = image
        self.x = x
        self.y = y
        self.transform = transform

    def __len__(self):
        return len((self.image))

    def __getitem__(self, idx):

        """
        patientID, SurvivalDuration, SurvivalCensor, Stage,
        ProgressionDuration, ProgressionCensor, MetaDuration, MetaCensor
        """
        transform = transforms.Compose([
                transforms.Resize(320),
                transforms.CenterCrop(299),
                transforms.ToTensor(),
                transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
                ])
        #device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        image = self.image[idx]
        x = self.x[idx]
        y = self.y[idx]
        image = image.convert('RGB')
        R = transform(image)

        sample = { 'image' : R,'X' : torch.tensor(x), 'Y' : torch.tensor(y) }
    
        return sample


def supernode_generation_from_h5(h5_file_path,  Argument, save_dir):
    """
    从预先提取的h5特征文件生成超级节点

    Args:
        h5_file_path: h5特征文件路径
        device: 计算设备
        Argument: 参数配置
        save_dir: 保存目录
    """
    import h5py

    if os.path.exists(save_dir) is False:
        os.mkdir(save_dir)

    origin_dir = os.path.join(save_dir, 'original')
    if os.path.exists(origin_dir) is False:
        os.mkdir(origin_dir)

    superpatch_dir = os.path.join(save_dir, 'superpatch')
    if os.path.exists(superpatch_dir) is False:
        os.mkdir(superpatch_dir)

    threshold = Argument.threshold
    spatial_threshold = Argument.spatial_threshold

    sample = os.path.basename(h5_file_path).split('.')[0]
    print(f"Processing sample: {sample}")

    # 读取h5文件中的特征和坐标
    try:
        with h5py.File(h5_file_path, 'r') as f:
            # 假设h5文件结构为：features, coords
            # 根据您的CLAM输出格式调整键名
            features = f['features'][:]  # shape: [N, feature_dim]
            coords = f['coords'][:]  # shape: [N, 2] (x, y coordinates)

            print(f"Loaded {features.shape[0]} patches with {features.shape[1]} features each")

    except Exception as e:
        print(f"Error reading h5 file {h5_file_path}: {e}")
        return 0

    # 将坐标和特征组合成DataFrame
    coordinate_df = pd.DataFrame({'X': coords[:, 0], 'Y': coords[:, 1]})
    feature_df = pd.DataFrame(features)

    # 合并坐标和特征
    graph_dataframe = pd.concat([coordinate_df, feature_df], axis=1)
    graph_dataframe = graph_dataframe.sort_values(by=['Y', 'X'])
    graph_dataframe = graph_dataframe.reset_index(drop=True)

    # 更新坐标DataFrame
    coordinate_df = graph_dataframe.iloc[:, 0:2]

    # 保存原始特征和坐标
    feature_df.to_csv(os.path.join(origin_dir, sample + '_feature_list.csv'), index=False)
    coordinate_df.to_csv(os.path.join(superpatch_dir, sample + '_node_location_list.csv'), index=False)

    # 添加原始索引
    index = list(graph_dataframe.index)
    graph_dataframe.insert(0, 'index_orig', index)

    # 超级节点生成与图构建
    node_dict = {}
    for i in range(len(coordinate_df)):
        node_dict.setdefault(i, [])

    X = max(coordinate_df['X'])
    Y = max(coordinate_df['Y'])

    print(f"Image dimensions: X={X}, Y={Y}")

    del feature_df

    # 网格分割参数
    gridNum = 4
    X_size = int(X / gridNum)
    Y_size = int(Y / gridNum)

    print("Building node connections...")
    with tqdm(total=(gridNum + 2) * (gridNum + 2)) as pbar:
        for p in range(gridNum + 2):
            for q in range(gridNum + 2):
                # 网格边界条件处理（保持原逻辑）
                if p == 0:
                    if q == 0:
                        is_X = graph_dataframe['X'] <= X_size * (p + 1)
                        is_X2 = graph_dataframe['X'] >= 0
                        is_Y = graph_dataframe['Y'] <= Y_size * (q + 1)
                        is_Y2 = graph_dataframe['Y'] >= 0
                        X_10 = graph_dataframe[is_X & is_Y & is_X2 & is_Y2]
                    elif q == (gridNum + 1):
                        is_X = graph_dataframe['X'] <= X_size * (p + 1)
                        is_X2 = graph_dataframe['X'] >= 0
                        is_Y = graph_dataframe['Y'] <= Y
                        is_Y2 = graph_dataframe['Y'] >= (Y_size * q - 2)
                        X_10 = graph_dataframe[is_X & is_Y & is_X2 & is_Y2]
                    else:
                        is_X = graph_dataframe['X'] <= X_size * (p + 1)
                        is_X2 = graph_dataframe['X'] >= 0
                        is_Y = graph_dataframe['Y'] <= Y_size * (q + 1)
                        is_Y2 = graph_dataframe['Y'] >= (Y_size * q - 2)
                        X_10 = graph_dataframe[is_X & is_Y & is_X2 & is_Y2]
                elif p == (gridNum + 1):
                    if q == 0:
                        is_X = graph_dataframe['X'] <= X
                        is_X2 = graph_dataframe['X'] >= (X_size * p - 2)
                        is_Y = graph_dataframe['Y'] <= Y_size * (q + 1)
                        is_Y2 = graph_dataframe['Y'] >= 0
                        X_10 = graph_dataframe[is_X & is_Y & is_X2 & is_Y2]
                    elif q == (gridNum + 1):
                        is_X = graph_dataframe['X'] <= X
                        is_X2 = graph_dataframe['X'] >= (X_size * p - 2)
                        is_Y = graph_dataframe['Y'] <= Y
                        is_Y2 = graph_dataframe['Y'] >= (Y_size * q - 2)
                        X_10 = graph_dataframe[is_X & is_Y & is_X2 & is_Y2]
                    else:
                        is_X = graph_dataframe['X'] <= X
                        is_X2 = graph_dataframe['X'] >= (X_size * p - 2)
                        is_Y = graph_dataframe['Y'] <= Y_size * (q + 1)
                        is_Y2 = graph_dataframe['Y'] >= (Y_size * q - 2)
                        X_10 = graph_dataframe[is_X & is_Y & is_X2 & is_Y2]
                else:
                    if q == 0:
                        is_X = graph_dataframe['X'] <= X_size * (p + 1)
                        is_X2 = graph_dataframe['X'] >= (X_size * p - 2)
                        is_Y = graph_dataframe['Y'] <= Y_size * (q + 1)
                        is_Y2 = graph_dataframe['Y'] >= 0
                        X_10 = graph_dataframe[is_X & is_Y & is_X2 & is_Y2]
                    elif q == (gridNum + 1):
                        is_X = graph_dataframe['X'] <= X_size * (p + 1)
                        is_X2 = graph_dataframe['X'] >= (X_size * p - 2)
                        is_Y = graph_dataframe['Y'] <= Y
                        is_Y2 = graph_dataframe['Y'] >= (Y_size * q - 2)
                        X_10 = graph_dataframe[is_X & is_Y & is_X2 & is_Y2]
                    else:
                        is_X = graph_dataframe['X'] <= X_size * (p + 1)
                        is_X2 = graph_dataframe['X'] >= (X_size * p - 2)
                        is_Y = graph_dataframe['Y'] <= Y_size * (q + 1)
                        is_Y2 = graph_dataframe['Y'] >= (Y_size * q - 2)
                        X_10 = graph_dataframe[is_X & is_Y & is_X2 & is_Y2]

                # 处理空网格
                if len(X_10) == 0:
                    pbar.update()
                    continue

                # 提取当前网格内的坐标和特征
                coordinate_dataframe = X_10.loc[:, ['X', 'Y']]
                X_10 = X_10.reset_index(drop=True)
                coordinate_list = coordinate_dataframe.values.tolist()
                index_list = coordinate_dataframe.index.tolist()

                feature_dataframe = X_10[X_10.columns.difference(['index_orig', 'X', 'Y'])]
                feature_list = feature_dataframe.values.tolist()

                # 计算坐标距离矩阵和特征余弦相似度矩阵
                coordinate_matrix = euclidean_distances(coordinate_list, coordinate_list)
                coordinate_matrix = np.where(coordinate_matrix > 2.9, 0, 1)
                cosine_matrix = cosine_similarity(feature_list, feature_list)

                # 构建邻接矩阵：同时满足空间距离和特征相似度条件
                Adj_list = (coordinate_matrix == 1).astype(int) * (cosine_matrix >= threshold).astype(int)

                # 填充节点连接关系字典
                for c, item in enumerate(Adj_list):
                    for node_index in np.array(index_list)[item.astype('bool')]:
                        if node_index == index_list[c]:
                            pass
                        else:
                            node_dict[index_list[c]].append(node_index)

                pbar.update()

    # 保存节点字典
    with open(os.path.join(origin_dir, sample + '_node_dict.pkl'), "wb") as a_file:
        pickle.dump(node_dict, a_file)

    print("Selecting super nodes...")
    # 按连接数降序排序，筛选关键节点
    dict_len_list = [len(node_dict[i]) for i in range(len(node_dict))]
    arglist_strict = np.argsort(np.array(dict_len_list))[::-1]

    # 严格筛选，保留连接数多的节点，删除其邻居节点
    for arg_value in arglist_strict:
        if arg_value in node_dict.keys():
            for adj_item in node_dict[arg_value]:
                if adj_item in node_dict.keys():
                    node_dict.pop(adj_item)
                    arglist_strict = np.delete(arglist_strict, np.argwhere(arglist_strict == adj_item))

    # 去重处理连接关系
    for key_value in node_dict.keys():
        node_dict[key_value] = list(set(node_dict[key_value]))

    print(f"Selected {len(node_dict)} super nodes from {len(coordinate_df)} original nodes")

    # 收集超级节点的坐标和特征
    supernode_coordinate_x_strict = []
    supernode_coordinate_y_strict = []
    supernode_feature_strict = []

    whole_feature = graph_dataframe[graph_dataframe.columns.difference(['index_orig', 'X', 'Y'])]

    print("Computing super node features...")
    with tqdm(total=len(node_dict.keys())) as pbar_node:
        for key_value in node_dict.keys():
            supernode_coordinate_x_strict.append(graph_dataframe['X'][key_value])
            supernode_coordinate_y_strict.append(graph_dataframe['Y'][key_value])

            # 处理特征：如果没有连接节点则使用自身特征，否则使用邻居和自身的平均特征
            if len(node_dict[key_value]) == 0:
                select_feature = whole_feature.iloc[key_value]
            else:
                select_feature = whole_feature.iloc[node_dict[key_value] + [key_value]]
                select_feature = select_feature.mean()

            # 拼接特征数据
            if len(supernode_feature_strict) == 0:
                temp_select = np.array(select_feature)
                supernode_feature_strict = np.reshape(temp_select, (1, len(temp_select)))
            else:
                temp_select = np.array(select_feature)
                supernode_feature_strict = np.concatenate(
                    (supernode_feature_strict, np.reshape(temp_select, (1, len(temp_select)))), axis=0)
            pbar_node.update()

    # 构建超级节点间的图结构
    print("Building super node graph...")
    coordinate_integrate = pd.DataFrame({
        'X': supernode_coordinate_x_strict,
        'Y': supernode_coordinate_y_strict
    })
    coordinate_matrix1 = euclidean_distances(coordinate_integrate, coordinate_integrate)
    coordinate_matrix1 = np.where(coordinate_matrix1 > spatial_threshold, 0, 1)

    fromlist = []
    tolist = []

    with tqdm(total=len(coordinate_matrix1)) as pbar_pytorch_geom:
        for i in range(len(coordinate_matrix1)):
            temp = coordinate_matrix1[i, :]
            selectindex = np.where(temp > 0)[0].tolist()
            for index in selectindex:
                fromlist.append(int(i))
                tolist.append(int(index))
            pbar_pytorch_geom.update()

    # 创建PyTorch Geometric数据对象
    edge_index = torch.tensor([fromlist, tolist], dtype=torch.long)
    x = torch.tensor(supernode_feature_strict, dtype=torch.float)

    # 添加位置信息
    pos = torch.tensor(np.column_stack((supernode_coordinate_x_strict, supernode_coordinate_y_strict)),
                       dtype=torch.float)
    data = Data(x=x, edge_index=edge_index, pos=pos)

    # 保存结果
    node_dict_df = pd.DataFrame.from_dict(node_dict, orient='index')
    node_dict_df.to_csv(os.path.join(superpatch_dir, sample + '_' + str(threshold) + '.csv'))
    torch.save(data, os.path.join(superpatch_dir, sample + '_' + str(threshold) + '_graph_torch.pt'))

    print(f"Graph saved with {data.x.shape[0]} nodes and {data.edge_index.shape[1]} edges")
    return 1


# 使用示例函数
def process_h5_files(h5_dir, device, Argument, save_dir):
    """
    批量处理h5文件
    """
    h5_files = [f for f in os.listdir(h5_dir) if f.endswith('.h5')]

    for h5_file in tqdm(h5_files, desc="Processing h5 files"):
        h5_path = os.path.join(h5_dir, h5_file)
        sample_save_dir = os.path.join(save_dir, h5_file.split('.')[0])

        try:
            supernode_generation_from_h5(h5_path, device, Argument, sample_save_dir)
            print(f"Successfully processed {h5_file}")
        except Exception as e:
            print(f"Error processing {h5_file}: {e}")
def Parser_main():

    parser = argparse.ArgumentParser(description="superpatch generation")
    parser.add_argument("--database", default='TCGA', help="Use in the savedir", type = str)
    parser.add_argument("--cancertype",default='BRCA',help="cancer type",type=str)
    parser.add_argument("--graphdir",default="/Sample_data_for_demo/Graph_test/",help="graph save dir",type=str)
    parser.add_argument("--imagedir",default="/data/BRCA/TS/",help="svs file location",type=str)
    parser.add_argument("--weight_path",default=None,help="pretrained weight path",type=str)
    parser.add_argument("--imagesize", default = 256, help ="crop image size", type = int)
    parser.add_argument("--threshold", default = 0.75, help = "cosine similarity threshold", type = float)
    parser.add_argument("--spatial_threshold", default = 5.5, help = "spatial threshold", type = float)
    parser.add_argument("--gpu", default = '0' , help = "gpu device number", type = str)
    return parser.parse_args()
def should_skip_processing(sample, base_dir):
        """判断当前样本是否已经处理完成，存在三个输出文件即认为已完成"""
        Argument = Parser_main()
        origin_dir = os.path.join(base_dir, 'original')
        superpatch_dir = os.path.join(base_dir, 'superpatch')

        file1 = os.path.join(origin_dir, f"{sample}_feature_list.csv")
        file2 = os.path.join(superpatch_dir, f"{sample}_node_location_list.csv")
        file3 = os.path.join(superpatch_dir, f"{sample}_{str(Argument.threshold)}_graph_torch.pt")

        return os.path.exists(file1) and os.path.exists(file2) and os.path.exists(file3)
def main():

    Argument = Parser_main()
    cancer_type = Argument.cancertype
    database = Argument.database
    image_dir = Argument.imagedir
    save_dir = Argument.graphdir
    gpu = Argument.gpu
    files = os.listdir(image_dir)

    if os.path.exists(save_dir) is False:
        os.mkdir(save_dir)
    save_dir = os.path.join(save_dir, database)
    if os.path.exists(save_dir) is False:
        os.mkdir(save_dir)
    save_dir = os.path.join(save_dir, cancer_type)

    final_files = [os.path.join(image_dir, file) for file in files]
    final_files.sort(key=lambda f: os.stat(f).st_size, reverse=False)
    #模型加载与初始化
    device = torch.device(int(gpu) if torch.cuda.is_available() else "cpu")
    model_ft = EfficientNet.from_pretrained('resnet-50', num_classes = 2)
    if Argument.weight_path is not None:
        weight_path = Argument.weight_path
        load_weight = torch.load(weight_path, map_location = device)
        model_ft.load_state_dict(load_weight)

    model_ft = model_ft.to(device)
    model_ft.eval()
    #图像数据处理与超级节点生成

    print("save",save_dir)
    # # 处理所有图像样本
    with tqdm(total=len(final_files)) as pbar_tot:
        for image in final_files:
            sample = os.path.splitext(os.path.basename(image))[0].split('.')[0]
            if should_skip_processing(sample, save_dir):
                print("save", save_dir)
                print(f"[跳过] {sample} 已处理完毕")
                pbar_tot.update()
                continue
            supernode_generation(image, model_ft, device, Argument, save_dir)
            pbar_tot.update()

    # 最后进行图过滤
    false_graph_filtering(4.3,root_dir,origin_file_dir)
    print("全部完成!")
if __name__ == "__main__":
    main()