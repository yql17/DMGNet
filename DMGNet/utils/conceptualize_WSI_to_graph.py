import os
import warnings
warnings.filterwarnings("ignore")
import feather
import pandas as pd
import scanpy as sc
import anndata as ad
import numpy as np
import random


from scipy.sparse import csr_matrix
"""CPU版：scanpy的UMAP降维后kNN
的区别
graph是后构图的
"""
#通过设置随机种子，确保代码的可重复性。
def setup_seed(seed):

    np.random.seed(seed)
    random.seed(seed)

setup_seed(42)
#这个函数用于判断两个 patch 是否是物理上的邻居。如果两个 patch 的曼坐标在水平或垂直方向上相邻（哈顿距离为1），则返回 True。
#
def are_neighbors(coord1, coord2):
    x1, y1 = coord1
    x2, y2 = coord2
    return abs(x1 - x2) + abs(y1 - y2) == 1


def construction_graph(clinical_data_path, slides_feature_path, cohort):
    # 读取临床数据
    if not os.path.exists(clinical_data_path):
        raise FileNotFoundError(f"Clinical data file {clinical_data_path} not found.")
    codebook = pd.read_csv(clinical_data_path)
    #codebook = pd.read_csv(f'{clinical_data_path}')
    slides_patch = []
    slides_patch_feature = []
    slides_edge = []
    slides_name = codebook.loc[:, 'sample'].to_list()
    count = 0
    for slide_name in slides_name:
        count = count + 1
        print(f'{count} {slide_name}')

        # 提取临床文件中slides列名的固定长度部分
        key_part = slide_name[:15]  # 这里假设固定长度为15，可按需调整
        slide_path = None

        # 遍历.feather文件目录，查找匹配的文件
        for root, dirs, files in os.walk(slides_feature_path):
            for file in files:
                if file.endswith('.feather'):
                    # 提取.feather文件名的固定长度部分
                    file_key_part = file[:15]  # 这里假设固定长度为15，可按需调整
                    if key_part == file_key_part:
                        slide_path = os.path.join(root, file)
                        break
            if slide_path:
                break

        if not slide_path:
            print(f"未找到匹配的.feather文件: {slide_name}")
            continue

        slide = feather.read_dataframe(slide_path)
        # 服务器运行用这个
        #slide = pd.read_feather(slide_path)
        slide_patches_features = slide.iloc[:, 0:].values
        slide_patches = list(slide.index)
        patches_coordinates = [(int(patch.split('-')[-2].split('_')[-1]), int(patch.split('-')[-1].split('.')[0])) for
                               patch in slide_patches]

        # Compute physics_edge, 计算物理邻接关系
        physics_edge = np.zeros((len(slide_patches), len(slide_patches)))
        for i in range(len(physics_edge)):
            for j in range(len(physics_edge)):
                coord1 = patches_coordinates[i]
                coord2 = patches_coordinates[j]
                if are_neighbors(coord1, coord2):
                    physics_edge[i][j] = 1

        # Compute logical_edge, 计算逻辑相似性关系,k近邻关系构建逻辑图
        obs = pd.DataFrame()
        obs['patches'] = slide_patches
        #var = [i for i in range(2048)]
        var = [i for i in range(len(slide_patches_features[0]))]
        var = pd.DataFrame(index=var)
        X = np.array(slide_patches_features)
        adata = ad.AnnData(X, obs=obs, var=var)
        n_neighbors = 9
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, method='umap', use_rep='X')
        logical_edge = adata.obsp['distances']
        logical_edge = logical_edge.toarray()
        logical_edge[logical_edge!=0] = 1
        #合并物理和逻辑关系
        adj_matrix = physics_edge + logical_edge
        adj_matrix[adj_matrix!=0] = 1
        adj_matrix = csr_matrix(adj_matrix)
        slides_patch.append(slide_patches)
        slides_patch_feature.append(slide_patches_features)
        slides_edge.append(adj_matrix)

    np.save(f'../../datasets/{cohort}/train35_patches_name.npy', np.array(slides_patch, dtype=object))
    np.save(f'../../datasets/{cohort}/train35_nodes.npy', np.array(slides_patch_feature, dtype=object))
    np.save(f'../../datasets/{cohort}/train35_edges.npy', np.array(slides_edge))


if __name__ == '__main__':
    # 定义路径和数据集名称
    clinical_data_path = "../../datasets/LUAD-datasets/train-LUADSur.csv" #  临床数据路径
    #slides_feature_path = "../../datasets/feature_test02"      # patch 特征路径
    slides_feature_path = "../../datasets/feature_normaled_35"
    cohort = "features_graph"                                  # 数据集名称

    # 调用函数
    construction_graph(clinical_data_path, slides_feature_path, cohort)