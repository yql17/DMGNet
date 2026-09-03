
library(dplyr)
# 1.读取数据-------------------
## 1.1读取count数值
library(tidyverse)
counts1<-fread("./TCGA-LUAD.star_counts.tsv.gz")
 ## 1.3生存数据读取                                                                                                                                                                                               
survial_data<- read_tsv("TCGA-LUAD.survival.tsv")
## 1.4 读取临床表征数据
clinical_data <- fread("./TCGA-LUAD.survival.tsv.gz")

# 2.过滤数据--------
## 2.1 过滤掉没有生存数据或者没有表达数据的样本
expr_sample <-colnames(counts2)
sur_sample <- survial_data$sample
view(expr_sample)
view(sur_sample)
#取交集
valid_sample <-intersect(expr_sample,sur_sample)
counts3<- counts2[,valid_sample]
#将列名变为行名
surv_dat1 <- column_to_rownames(survial_data,"sample")
#过滤样本
surv_dat1 <- surv_dat1[valid_sample,]

## 2.2 过滤非肿瘤样本
sample_type <- str_split(colnames(counts3),pattern ='-',n=4,simplify = TRUE)#分割count3的列名为四部分
sample_type <- as.data.frame(sample_type)#转化成dataframe
sample_type <- sample_type[,"V4"]
unique(sample_type)#查看样本类型
sample_group <- cbind(sample_type)
#sample_grop[,1] <- ifelse(sample_grop[,1]=="11A","normal","tumor")
sample_group <- ifelse(sample_group[, 1] %in% c("11A", "11B"), "normal", "tumor")


view(sample_group)
sample_group <- data.frame(Group = sample_group)
rows <- sample_group[,1]=="tumor"
view(rows)
counts4 <- counts3[,rows]
surv_dat2 <- surv_dat1[rows,]#过滤后的样本

# 3.数据转化-----
##3.1 获取所有基因的ID。并转换成为基因名
gene_id <- rownames(counts4)
#去除基因ID的.
gene_id <- str_split(gene_id,"[.]",simplify = T)[,1]
options(BioC_mirror = "https://mirrors.tuna.tsinghua.edu.cn/bioconductor")
library(org.Hs.eg.db)
gene_name <- mapIds(org.Hs.eg.db,gene_id1,"SYMBOL","ENSEMBL")
view(gene_name)
counts5 <- counts4
rownames(counts5) <- gene_id1
counts5 <- rownames_to_column(counts5,"gene_id")
gene_name <- cbind(gene_name)
gene_name <- as.data.frame(gene_name)
gene_name <- rownames_to_column(gene_name,"gene_id")
counts6 <- left_join(counts5,gene_name,by="gene_id")
counts6 <- relocate(counts6,gene_name,.after = gene_id)
counts6 <- column_to_rownames(counts6,"gene_id")
counts7 <- aggregate(.-gene_name,FUM=mean,data=counts6)
counts8 <- column_to_rownames(counts7,"gene_name")
