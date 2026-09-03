# -*- coding:utf8 -*-
# -*- coding:utf8 -*-

import os
import copy
import random

import joblib
import torch
import torch_geometric.transforms as T
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch_geometric.loader import DataLoader  # 适用于批量处理 `Data` 类型图

from torch import optim, nn
from torch_geometric.transforms import Polar
from torch_geometric.data import DataListLoader
from torch_geometric.nn import DataParallel
from torch_geometric.data import Data
from torch_geometric.data import Dataset
from torch.optim.lr_scheduler import OneCycleLR

from tqdm import tqdm

# from Gene_project.Gene import SNN_GAT, generate_graph_data, EarlyStopping
from Superpatch_network_construction.SNN_cosline_similarity import SNN_GAT, generate_graph_data, EarlyStopping
from Superpatch_network_construction.utils.util import CoxLoss
from model_selection import model_selection
from utils import train_test_split, train_test_split_from_csv, TrainValid_path, cox_sort2, train_test_split3, \
    cross_validation_split
from utils import makecheckpoint_dir_graph as mcd
from utils import non_decay_filter
from utils import coxph_loss
from utils import cox_sort
from utils import accuracytest

from torch.utils.data.sampler import Sampler
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="PyDev debugger: warning: trying to add breakpoint.*")
class Sampler_custom(Sampler):

    def __init__(self, event_list, censor_list, batch_size):
        self.event_list = event_list
        self.censor_list = censor_list
        self.batch_size = batch_size

    def __iter__(self):
        train_batch_sampler = []
        Event_idx = copy.deepcopy(self.event_list)
        Censored_idx = copy.deepcopy(self.censor_list)
        np.random.shuffle(Event_idx)
        np.random.shuffle(Censored_idx)

        Int_event_batch_num = (Event_idx.shape[0] // 2) * 2
        Event_idx = Event_idx[:Int_event_batch_num]
        Int_censor_batch_num = Censored_idx.shape[0] // (self.batch_size - 2) * (self.batch_size - 2)
        Censored_idx = Censored_idx[:Int_censor_batch_num]

        Event_idx_selected = Event_idx.reshape(-1, 2)
        Censored_idx_selected = Censored_idx.reshape(-1, self.batch_size - 2)

        min_len = min(Event_idx_selected.shape[0], Censored_idx_selected.shape[0])
        Event_idx_selected = Event_idx_selected[:min_len]
        Censored_idx_selected = Censored_idx_selected[:min_len]

        for c in range(min_len):
            train_batch_sampler.append(
                Event_idx_selected[c].tolist() + Censored_idx_selected[c].tolist()
            )
        return iter(train_batch_sampler)

    def __len__(self):
        return len(self.event_list) // 2

class CoxGraphDataset(Dataset):
    def __init__(self, filelist, survlist, stagelist, censorlist, Metadata, mode, model, transform=None, pre_transform=None):
        super(CoxGraphDataset, self).__init__()
        self.filelist = filelist
        self.survlist = survlist
        self.stagelist = stagelist
        self.censorlist = censorlist
        self.Metadata = Metadata
        self.mode = mode
        self.model = model
        self.polar_transform = Polar()

    def processed_file_names(self):
        return self.filelist

    def len(self):
        return len(self.filelist)

    def get(self, idx):
        data_origin = torch.load(self.filelist[idx])
        transfer = T.ToSparseTensor()
        item = self.filelist[idx].split('/')[-1].split('.pt')[0].split('_')[0]
        mets_class = 0

        survival = self.survlist[idx]
        phase = self.censorlist[idx]
        stage = self.stagelist[idx]

        data_re = Data(x=data_origin.x[:,:1792], edge_index=data_origin.edge_index)
        mock_data = Data(x=data_origin.x[:,:1792], edge_index=data_origin.edge_index, pos=data_origin.pos)
        data_re.pos = data_origin.pos
        data_re_polar = self.polar_transform(mock_data)
        polar_edge_attr = data_re_polar.edge_attr

        if data_re.edge_index.shape[1] != data_origin.edge_attr.shape[0]:
            print(f'Error in {self.filelist[idx].split("/")[-1]}: edge index and attr mismatch')
        data = transfer(data_re)
        data.survival = torch.tensor(survival)
        data.phase = torch.tensor(phase)
        data.mets_class = torch.tensor(mets_class)
        data.stage = torch.tensor(stage)
        data.item = item
        data.edge_attr = polar_edge_attr
        data.pos = data_origin.pos
        return data

def extract_sample_id(file_path,DataType):
    file_name = os.path.basename(file_path)
    id_part = file_name.split('_')[0]
    if DataType=='LUAD':
        sample_id = '-'.join(id_part.split('-')[:3])  # 提取核心样本ID（如TCGA-MP-A5C7）
    else:
        sample_id = '-'.join(id_part.split('-')[:4])  # 提取核心样本ID（如TCGA-MP-A5C7-01A）
    return sample_id

def save_gene_dataset(gene_df, gene_id_col, wsi_ids, save_dir, save_name):
    matched_gene = gene_df[gene_df[gene_id_col].isin(wsi_ids)].copy()
    matched_gene['id_order'] = matched_gene[gene_id_col].apply(lambda x: wsi_ids.index(x))
    matched_gene = matched_gene.sort_values('id_order').drop(columns='id_order')
    save_path = os.path.join(save_dir, save_name)
    matched_gene.to_csv(save_path, index=False)
    print(f"已保存基因数据集：{save_path}，样本数：{len(matched_gene)}")
    return save_path
def filter_and_keep_order(ids_list, valid_set):
    """过滤列表并保持原始顺序"""
    return [id_ for id_ in ids_list if id_ in valid_set]
def set_seed(seed=12345):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
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

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.alpha = nn.Parameter(torch.tensor([0.5], dtype=torch.float32))  # 可学习参数

    def forward(self, x1, x2):
        # 假设 alpha 是用于融合两个风险值
        alpha = torch.sigmoid(self.alpha).to(x1.device)
        risk = (1 - alpha) * x1 + alpha * x2
        return risk

def Train(Argument):
    set_seed()
    n_genes = 3000
    hidden_dim1 = 64
    output_dim = 16
    num_class = 1
    checkpoint_dir, Figure_dir = mcd(Argument)
    batch_num = int(Argument.batch_size)
    device = torch.device(f"cuda:{Argument.gpu}" if torch.cuda.is_available() else "cpu")

    # 加载临床和图数据
    metadata_path = './Sample_data_for_demo/%s_data/top3000_clinical_%s.csv' % (Argument.DatasetType, Argument.DatasetType)
    Metadata = pd.read_csv(metadata_path)

    # 划分WSI数据集并提取样本ID
    TrainRoot = TrainValid_path(Argument.DatasetType)
    print("路径",TrainRoot)
    Trainlist = [item for item in os.listdir(TrainRoot) if '0.75_graph_torch_4.3_artifact_sophis_final.pt' in item]
    Fi = Argument.FF_number
    Train_set,Test_set = cross_validation_split(Trainlist, Metadata, Argument.DatasetType, TrainRoot, Fi)
    # === 基因表达矩阵 & 生存数据 ===
    gene_path = './Sample_data_for_demo/%s_data/top3000_genes_%s.csv' % (Argument.DatasetType,Argument.DatasetType)
    gene_df = pd.read_csv(gene_path)
    sur_path='./Sample_data_for_demo/%s_data/top3000_survival_%s.csv' % (Argument.DatasetType,Argument.DatasetType)
    sur_df = pd.read_csv(sur_path)

    # ===================== 关键：样本ID对齐与排序 =====================
    # 1. 提取WSI样本ID（保持原始顺序）
    train_wsi_ids_ordered = [extract_sample_id(f,Argument.DatasetType) for f in Train_set[0]]
    test_wsi_ids_ordered = [extract_sample_id(f,Argument.DatasetType) for f in Test_set[0]]
    all_wsi_ids = set(train_wsi_ids_ordered + test_wsi_ids_ordered)

    # 2. 计算三者交集
    all_gene_ids = set(gene_df["submitter_id"])
    all_sur_ids = set(sur_df["submitter_id"])
    valid_ids = all_wsi_ids & all_gene_ids & all_sur_ids
    print(f"三者共有的有效样本数量: {len(valid_ids)}")

    # 3. 过滤WSI ID并保持顺序
    train_wsi_ids_filtered = filter_and_keep_order(train_wsi_ids_ordered, valid_ids)

    test_wsi_ids_filtered = filter_and_keep_order(test_wsi_ids_ordered, valid_ids)

    # 4. 基因数据按WSI顺序对齐
    gene_df_filtered = gene_df[gene_df["submitter_id"].isin(valid_ids)].set_index("submitter_id")
    train_gene = gene_df_filtered.reindex(train_wsi_ids_filtered)
    test_gene = gene_df_filtered.reindex(test_wsi_ids_filtered)

    # 5. 生存数据按WSI顺序对齐
    sur_df_filtered = sur_df[sur_df["submitter_id"].isin(valid_ids)].set_index("submitter_id")
    train_sur = sur_df_filtered.reindex(train_wsi_ids_filtered)

    test_sur = sur_df_filtered.reindex(test_wsi_ids_filtered)

    # 6. 验证对齐结果（必须全部为True）
    print("\n=== 对齐验证 ===")
    print(f"训练集基因与WSI ID对齐: {all(train_gene.index == train_wsi_ids_filtered)}")
    print(f"测试集基因与WSI ID对齐: {all(test_gene.index == test_wsi_ids_filtered)}")
    print(f"训练集生存数据与WSI ID对齐: {all(train_sur.index == train_wsi_ids_filtered)}")

    # 处理可能的空值
    train_gene = train_gene.fillna(0)
    test_gene = test_gene.fillna(0)


    # === 拼合 all_data, all_sur ===
    all_data = pd.concat([train_gene,test_gene])
    all_sur = pd.concat([train_sur, test_sur])
    N = len(all_data)
    T = len(train_gene)  # 训练集样本数

    train_mask = torch.zeros(N, dtype=torch.bool, device=device)

    test_mask = torch.zeros(N, dtype=torch.bool, device=device)
    train_mask[:T] = True  # 前T个样本：训练集
    test_mask[T:] = True  # 剩余样本：测试集

    mask_path = os.path.join('Sample_data_for_demo/test', "masks.pt")
    torch.save({
        "train_mask": train_mask.cpu(),
        "test_mask": test_mask.cpu()
    }, mask_path)
    print(f"已保存训练/测试mask至 {mask_path}")

    # 提取训练/测试WSI样本ID（与SNN样本ID一一对应）
    train_wsi_ids = [extract_sample_id(file_path,Argument.DatasetType) for file_path in Train_set[0]]

    test_wsi_ids = [extract_sample_id(file_path,Argument.DatasetType) for file_path in Test_set[0]]
    print(f"训练集WSI样本数：{len(train_wsi_ids)},测试集WSI样本数：{len(test_wsi_ids)}")
    early_stopping = EarlyStopping(patience=10, verbose=True)
    # 创建数据集和数据加载器
    TestDataset = CoxGraphDataset(
        filelist=Test_set[0], survlist=Test_set[1], stagelist=Test_set[3], censorlist=Test_set[2],
        Metadata=Metadata, mode=Argument.DatasetType, model=Argument.model
    )
    TrainDataset = CoxGraphDataset(
        filelist=Train_set[0], survlist=Train_set[1], stagelist=Train_set[3], censorlist=Train_set[2],
        Metadata=Metadata, mode=Argument.DatasetType, model=Argument.model
    )
    # 自定义采样器（平衡事件/删失样本）
    Event_idx = np.where(np.array(Train_set[2]) == 1)[0]
    Censored_idx = np.where(np.array(Train_set[2]) == 0)[0]
    train_batch_sampler = Sampler_custom(Event_idx, Censored_idx, batch_num)

    # 数据加载器
    g = torch.Generator()
    g.manual_seed(12345)
    test_loader = DataListLoader(TestDataset, batch_size=batch_num, shuffle=True, num_workers=0, pin_memory=True, drop_last=False)#,generator=g
    train_loader = DataListLoader(TrainDataset, batch_sampler=train_batch_sampler, num_workers=0, pin_memory=True)#,generator=g
    # 初始化模型
    # 1. GNN模型（病理图像分支）
    gnn_model = model_selection(Argument)
    model_parameter_groups = non_decay_filter(gnn_model)
    gnn_model = DataParallel(gnn_model, device_ids=[2, 5], output_device=2)
    gnn_model = gnn_model.to(device)
    # 2. SNN模型（基因分支）
    snn_model = SNN_GAT(nfeat=n_genes, nhid1=hidden_dim1, output=output_dim, num_class=num_class)
    scaler = StandardScaler()
    scaled_df = pd.DataFrame(scaler.fit_transform(all_data.values),
                             columns=all_data.columns, index=all_data.index)

    ##########################
    # 定义基因图保存路径
    G_path="Sample_data_for_demo/LUAD_data"
    gene_graph_path = os.path.join(G_path, "gene_graph_data.pt")
    gene_features, gene_adj = generate_graph_data(scaled_df)
    # 保存基因图数据（特征和邻接矩阵）
    torch.save(
        {"features": gene_features, "adj": gene_adj},
        gene_graph_path
    )
    print(f"已保存基因图数据至: {gene_graph_path}")

    # 将基因图数据移至设备
    gene_features = gene_features.to(device)
    gene_adj = gene_adj.to(device)
    # print("特征：",gene_features)
    # print("边：",gene_adj)
    ###############################
    # gene_features, gene_adj = generate_graph_data(scaled_df)  # 基因图数据（特征+邻接矩阵）
    snn_model = snn_model.to(device)
    model_all=MyModel()
    # 优化器和损失函数
    optimizer_snn = optim.Adam(snn_model.parameters(), lr=0.001, weight_decay=0.1)
    optimizer_gnn = optim.Adam(model_parameter_groups, lr=Argument.learning_rate, weight_decay=Argument.weight_decay)
    scheduler_gnn = OneCycleLR(optimizer_gnn, max_lr=Argument.learning_rate, steps_per_epoch=len(train_loader),
                           epochs=Argument.num_epochs)

    optimizer_alpha = optim.Adam(model_all.parameters(), lr=Argument.learning_rate, weight_decay=Argument.weight_decay)
    scheduler_alpha = OneCycleLR(optimizer_alpha, max_lr=0.001, steps_per_epoch=len(train_loader),
                               epochs=Argument.num_epochs)
    cox_loss = coxph_loss().to(device)
    # 训练过程变量
    bestloss = float('inf')

    bestacc = 0.0

    bestepoch = 0

    loader = {'train': train_loader, 'test': test_loader}
    BestAccDict = {'train': 0, 'val': 0, 'test':0}
    FFCV_accuracy = []
    FFCV_best_epoch = []
    test_metric_rows = []#新增：存储每个 epoch 的 TEST 指标
    # 训练主循环
    with tqdm(total=Argument.num_epochs) as pbar:
        for epoch in range(int(Argument.num_epochs)):
            for mode in ['train', 'test']:
                # 设置模型模式（训练/评估）
                if mode == 'train':
                    gnn_model.train()
                    snn_model.train()
                    model_all.train()
                    grad_flag = True
                else:
                    gnn_model.eval()
                    snn_model.eval()
                    model_all.eval()

                    grad_flag = False

                with torch.set_grad_enabled(grad_flag):
                    EpochSurv_snn, EpochPhase_snn, EpochRisk_snn, EpochID_snn, EpochStage_snn = [], [], [],[],[]
                    EpochSurv_gat, EpochPhase_gat, EpochRisk_gat, EpochID_gat, EpochStage_gat,EpochID_gat,sort_idx_snn = [], [], [],[],[],[],[]
                    EpochSNN_risk=[]
                    Epochloss_gat = 0.0
                    Epochloss_snn = 0.0
                    Epochloss_all = 0.0
                    batchcounter = 1
                    pass_count = 0
                    SNN_risk, _ = snn_model(gene_features, gene_adj)
                    SNN_risk = SNN_risk.view(-1)
                    if mode=="train":
                        ######基因~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                        # print('train:', gene_features)
                        optimizer_snn.zero_grad()

                        # print('risk:', SNN_risk)
                        # print('mask:', train_mask)
                        train_preds = SNN_risk[train_mask].to(device)
                        train_preds = train_preds.unsqueeze(1)
                        snn_risklist=train_preds
                        OS_train = all_sur['OS'][train_mask.cpu().numpy()]
                        Time_train = all_sur['OS.time'][train_mask.cpu().numpy()]
                        all_sur["submitter_id"] = all_sur.index

                        Gene_ID = all_sur['submitter_id'][train_mask.cpu().numpy()]

                        Time_train = torch.tensor(Time_train.values, device=device)
                        OS_train = torch.tensor(OS_train.values, dtype=torch.float32, device=device)

                        snn_loss = CoxLoss(Time_train, OS_train, train_preds)

                        EpochPhase_snn = OS_train
                        #Epochacc_snn = accuracytest(Time_train, train_preds,OS_train)
                        snn_loss.backward()
                        optimizer_snn.step()

                        with torch.no_grad():
                            preds = SNN_risk[test_mask].to(device)
                            preds = preds.unsqueeze(1)
                            snn_risklist = preds
                            # print("test_preds:",test_preds)
                            OS= all_sur['OS'][test_mask.cpu().numpy()]
                            Time = all_sur['OS.time'][test_mask.cpu().numpy()]
                            all_sur["submitter_id"] = all_sur.index
                            Gene_ID = all_sur['submitter_id'][test_mask.cpu().numpy()]
                            Time = torch.tensor(Time.values, dtype=torch.float32, device=device)
                            OS= torch.tensor(OS.values, dtype=torch.float32, device=device)


                            EpochPhase_snn = OS
                            snn_loss = CoxLoss(Time, OS, preds)
                            #Epochacc_snn = accuracytest(Time, preds,OS)

                    for c, d in enumerate(loader[mode], 1):
                        #d = d.to(device)
                        optimizer_gnn.zero_grad()

                        # 提取批次样本信息
                        tempsurvival = torch.tensor([data.survival for data in d])
                        tempphase = torch.tensor([data.phase for data in d])
                        tempID = np.asarray([data.item for data in d])
                        tempstage = torch.tensor([data.stage for data in d])
                        tempmeta = torch.tensor([data.mets_class for data in d])
                        if Argument.DatasetType=='LUAD':
                            sampleID = np.asarray(['-'.join(data.item.split('-')[:3]) for data in d])  # 样本ID列表
                        else:
                            sampleID = np.asarray(['-'.join(data.item.split('-')[:4]) for data in d])  # 样本ID列表


                    ####WSI~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

                        # 多模态模型前向传播（融合风险值）
                        gat_risk,_=gnn_model(d)
                        #gat_risk=gat_risk.view(-1).clone()

                        gat_risklist,tempsurvival,tempphase, tempmeta,EpochSurv_gat,EpochPhase_gat,EpochRisk_gat,EpochStage_gat,EpochID_gat =cox_sort2(gat_risk,tempsurvival,tempphase,tempmeta, tempstage, sampleID, EpochSurv_gat, EpochPhase_gat, EpochRisk_gat, EpochStage_gat, EpochID_gat)

                        if torch.sum(tempphase).cpu().item() < 1:
                            pass_count += 1

                        else:

                            #gat_loss = cox_loss(gat_risklist, tempsurvival, tempphase)
                            gat_loss = CoxLoss(tempsurvival, tempphase, gat_risklist)
                            if mode == 'train':
                                gat_loss.backward()
                                torch.nn.utils.clip_grad_norm_(model_parameter_groups[0]['params'],
                                                               max_norm=Argument.clip_grad_norm_value,
                                                               error_if_nonfinite=True)
                                torch.nn.utils.clip_grad_norm_(model_parameter_groups[1]['params'],
                                                               max_norm=Argument.clip_grad_norm_value,
                                                               error_if_nonfinite=True)

                                optimizer_gnn.step()
                                scheduler_gnn.step()


                        Epochloss_gat += gat_loss.cpu().detach().item()
                        Epochloss_snn += snn_loss.cpu().detach().item()

                        # 记录损失和准确率
                        batchcounter += 1

                        optimizer_alpha.zero_grad()


                        # 转换为 numpy 方便索引

                        snn_ids = list(Gene_ID)
                        gat_ids = list(EpochID_gat)

                        # 构建索引映射（gat中的ID -> snn中的index）
                        id_to_snn_index = {id_: idx for idx, id_ in enumerate(snn_ids)}
                        matched_indices = [id_to_snn_index[id_] for id_ in gat_ids if id_ in id_to_snn_index]
                        EpochRisk_gat = torch.FloatTensor(EpochRisk_gat)
                        # 对齐后的 SNN 风险（按 GAT 顺序）
                        x1 = EpochRisk_gat.detach().clone()
                        x2 = snn_risklist.detach().clone()
                        x2 = x2[matched_indices]
                        x1 = torch.FloatTensor(x1).to(device)
                        OS_train=OS_train.detach().clone()
                        Time_train=Time_train.detach().clone()
                        # 对齐生存时间和事件（也要根据 GAT 顺序）
                        matched_surv = OS_train[matched_indices]
                        matched_time = Time_train[matched_indices]
                        EpochSNN_risk=snn_risklist[matched_indices]
                        EpochSNN_risk= torch.tensor(EpochSNN_risk,device=device)
                        EpochSurv_gat = torch.tensor(EpochSurv_gat,device=device)  # device 例如 "cuda:0"
                        EpochPhase_gat = torch.tensor(EpochPhase_gat, device=device)
                        EpochRisk_gat = torch.tensor(EpochRisk_gat, device=device)

                        EpochSurv_gat = EpochSurv_gat.detach().clone()
                        EpochPhase_gat=EpochPhase_gat.detach().clone()
                        EpochRisk_gat=EpochRisk_gat.detach().clone()

                        # 排序
                        sort_idx = torch.argsort(EpochSurv_gat, descending=True)
                        x1 = x1[sort_idx]
                        x2 = x2[sort_idx]

                        EpochRisk_gat = EpochRisk_gat[sort_idx]
                        EpochSurv_gat = EpochSurv_gat[sort_idx]
                        EpochPhase_gat = EpochPhase_gat[sort_idx]
                        matched_surv = matched_surv[sort_idx]
                        matched_time = matched_time[sort_idx]
                        EpochSNN_risk=EpochSNN_risk[sort_idx]
                        # 融合 & loss
                        x2 = x2.view(-1)
                        all_risk = model_all(x1, x2).unsqueeze(1)
                        all_loss = CoxLoss(EpochSurv_gat, EpochPhase_gat, all_risk)
                        all_loss.backward()
                        optimizer_alpha.step()
                        scheduler_alpha.step()

                    else:
                        snn_ids = list(Gene_ID)
                        gat_ids = list(EpochID_gat)

                        # 构建索引映射（gat中的ID -> snn中的index）
                        id_to_snn_index = {id_: idx for idx, id_ in enumerate(snn_ids)}
                        matched_indices = [id_to_snn_index[id_] for id_ in gat_ids if id_ in id_to_snn_index]
                        EpochRisk_gat = torch.FloatTensor(EpochRisk_gat)
                        # 对齐后的 SNN 风险（按 GAT 顺序）
                        x1 = EpochRisk_gat.detach().clone()
                        x2 = snn_risklist.detach().clone()
                        x2 = x2[matched_indices]
                        x1 = torch.FloatTensor(x1).to(device)
                        OS = OS.detach().clone()
                        Time = Time.detach().clone()
                        # 对齐生存时间和事件（也要根据 GAT 顺序）
                        matched_surv = OS[matched_indices]
                        matched_time = Time[matched_indices]
                        EpochSNN_risk = snn_risklist[matched_indices]
                        EpochSNN_risk = torch.tensor(EpochSNN_risk, device=device)
                        EpochSurv_gat = torch.tensor(EpochSurv_gat, device=device)  # device 例如 "cuda:0"
                        EpochPhase_gat = torch.tensor(EpochPhase_gat, device=device)
                        EpochRisk_gat = torch.tensor(EpochRisk_gat, device=device)
                        EpochSurv_gat = EpochSurv_gat.detach().clone()
                        EpochPhase_gat = EpochPhase_gat.detach().clone()
                        EpochRisk_gat = EpochRisk_gat.detach().clone()

                        # 排序
                        sort_idx = torch.argsort(EpochSurv_gat, descending=True)
                        x1 = x1[sort_idx]
                        x2 = x2[sort_idx]
                        EpochRisk_gat=EpochRisk_gat[sort_idx]
                        EpochSurv_gat = EpochSurv_gat[sort_idx]
                        EpochPhase_gat = EpochPhase_gat[sort_idx]
                        matched_surv = matched_surv[sort_idx]
                        matched_time = matched_time[sort_idx]
                        EpochSNN_risk = EpochSNN_risk[sort_idx]
                        # 融合 & loss
                        x2 = x2.view(-1)
                        all_risk = model_all(x1, x2).unsqueeze(1)
                        all_loss = CoxLoss(EpochSurv_gat, EpochPhase_gat, all_risk)

                        # 跳过无有效样本的epoch
                    Epochloss_all += all_loss.cpu().detach().item()
                    if len(EpochPhase_snn) < 1 or torch.sum(torch.tensor(EpochPhase_snn)) < 1:
                        print(f"[{mode}] 跳过轮次 {epoch}（无有效事件样本）")
                        continue


                    print(f"当前alpha值: {model_all.alpha.item():.4f}")

                    # 计算epoch级指标
                    Epochacc = accuracytest(torch.tensor(EpochSurv_gat), torch.tensor(all_risk), torch.tensor(EpochPhase_gat))
                    Epochacc_gat = accuracytest(torch.tensor(EpochSurv_gat), torch.tensor(EpochRisk_gat), torch.tensor(EpochPhase_gat))
                    #####新加#####
                    Epochacc_snn = accuracytest(torch.tensor(EpochSurv_gat), torch.tensor(EpochSNN_risk), torch.tensor(EpochPhase_gat))




                    gat_Epochloss_avg = Epochloss_gat / batchcounter
                    snn_Epochloss_avg = Epochloss_snn / batchcounter
                    all_Epochloss_avg = Epochloss_all / batchcounter
                    Epochloss=gat_Epochloss_avg+snn_Epochloss_avg+all_Epochloss_avg

                    # 更新最佳指标
                    if mode == 'train':
                        if Epochacc_snn > BestAccDict['train']:
                            BestAccDict['train'] = Epochacc_snn


                    elif mode == 'test':
                        if Epochacc_snn > BestAccDict['test']:
                            BestAccDict['test'] = Epochacc_snn

                    # 打印日志
                    print(f"\n轮次: {epoch} | 模式: {mode}")
                    print(f"gat损失: {gat_Epochloss_avg:.4f} |snn损失: {snn_Epochloss_avg:.4f} |融合损失: {all_Epochloss_avg:.4f} | C指数: {Epochacc:.4f} |C_gat指数: {Epochacc_gat:.4f} |C_snn指数: {Epochacc_snn:.4f} | 跳过批次: {pass_count}")

                    # 测试阶段保存最优模型
                    if mode == 'test':
                        test_metric_rows.append({
                            "epoch": int(epoch),
                            "gat_loss": float(gat_Epochloss_avg),
                            "snn_loss": float(snn_Epochloss_avg),
                            "all_loss": float(all_Epochloss_avg),
                            "c_index_gat": float(Epochacc_gat),
                            "c_index_snn": float(Epochacc_snn),
                            "c_index_all": float(Epochacc),
                        })
                        if (epoch == 0) or (Epochacc > bestacc) or (Epochloss < bestloss):
                            bestepoch = epoch
                            bestacc = max(bestacc, Epochacc)
                            bestloss = min(bestloss, Epochloss)
                            checkpoint_path = os.path.join(checkpoint_dir,
                                                               f"epoch_all-{epoch},acc-{Epochacc:.4f},loss-{Epochloss:.4f}.pt")
                            checkpoint_path_gnn = os.path.join(checkpoint_dir,
                                                               f"epoch_gat-{epoch},acc-{Epochacc_gat:.4f},loss-{gat_Epochloss_avg:.4f}.pt")
                            checkpoint_path_snn = os.path.join(checkpoint_dir,
                                                               f"epoch_snn-{epoch},acc-{Epochacc_snn:.4f},loss-{snn_Epochloss_avg:.4f}.pt")

                            torch.save(snn_model.state_dict(), checkpoint_path_snn)
                            torch.save(gnn_model.state_dict(), checkpoint_path_gnn)
                            torch.save(model_all.state_dict(), checkpoint_path)
                            print(f"保存最优模型至: {checkpoint_path}")



            pbar.update()

    # 交叉验证结果整理
            FFCV_accuracy.append(bestacc)
            FFCV_best_epoch.append(bestepoch)
            bestFi = np.argmax(FFCV_accuracy)
            best_checkpoint_dir = os.path.join(checkpoint_dir, str(bestFi))
            best_figure_dir = os.path.join(checkpoint_dir, str(bestFi))
            Argument.checkpoint_dir = best_checkpoint_dir

    return gnn_model,snn_model,model_all, best_checkpoint_dir, best_figure_dir, FFCV_best_epoch[bestFi]
