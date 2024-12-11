import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import laplacian
import hnswlib
import tqdm

# SPADE 所需的函数
def hnsw(features, k=10, ef=100, M=48):
    print("Initializing HNSW Index...")
    num_samples, dim = features.shape
    print(f"Number of samples: {num_samples}, Dimension: {dim}")
    p = hnswlib.Index(space='l2', dim=dim)
    p.init_index(max_elements=num_samples, ef_construction=ef, M=M)
    labels_index = np.arange(num_samples)
    p.add_items(features, labels_index)
    p.set_ef(ef)
    neighs, weight = p.knn_query(features, k + 1)
    print(f"Found nearest neighbors for k={k + 1}")
    return neighs, weight

def construct_adj(neighs, weight):
    print("Constructing adjacency matrix...")
    dim = neighs.shape[0]
    k = neighs.shape[1] - 1

    idx0 = np.arange(dim)
    row = np.repeat(idx0.reshape(-1,1), k, axis=1).reshape(-1,)
    col = neighs[:, 1:].reshape(-1,)
    all_row = np.concatenate((row, col), axis=0)
    all_col = np.concatenate((col, row), axis=0)
    data = np.ones(all_row.shape[0])
    adj = csr_matrix((data, (all_row, all_col)), shape=(dim, dim))
    print("Adjacency matrix constructed.")
    return adj

# 计算SPADE score
def spade_score(input_features, output_features, k=10, num_eigs=2):
    print("Calculating SPADE score...")
    # 构建输入和输出的邻接矩阵
    neighs_in, dist_in = hnsw(input_features, k)
    adj_in = construct_adj(neighs_in, dist_in)
    neighs_out, dist_out = hnsw(output_features, k)
    adj_out = construct_adj(neighs_out, dist_out)

    # 计算拉普拉斯矩阵
    print("Calculating Laplacian matrices...")
    L_in = laplacian(adj_in, normed=True)
    L_out = laplacian(adj_out, normed=True)

    # 计算广义特征值和特征向量
    print("Computing eigenvalues and eigenvectors...")
    eigvals_in, eigvecs_in = np.linalg.eig(L_in.toarray())
    eigvals_out, eigvecs_out = np.linalg.eig(L_out.toarray())

    # 选择最大特征值作为 SPADE 分数
    spade_score_value = max(eigvals_out) / max(eigvals_in)
    print(f"SPADE Score: {spade_score_value}")
    return spade_score_value





if __name__ == "__main__":
    # 设置设备
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 加载 ResNet50 模型
    print("Loading ResNet50 model...")
    resnet50 = models.resnet50(pretrained=True)
    resnet50.eval()
    resnet50 = resnet50.to(device)

    # 数据集和数据加载器（使用 CIFAR10 的 100 个数据样本）
    print("Loading CIFAR10 dataset...")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = CIFAR10(root='./data', train=False, transform=transform, download=True)
    subset = False
    if subset:
        subset = Subset(dataset, list(range(100)))  
        dataloader = DataLoader(subset, batch_size=100, shuffle=False)
    else:
        dataloader = DataLoader(dataset, batch_size=100, shuffle=False)

    input_features = []
    output_features = []

    print("Extracting input and output features...")
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            # 将图片展平并移动到CPU
            input_features.append(images.view(images.size(0), -1).cpu())
            
            
            # 提取特征并移动到CPU
            features = resnet50(images).cpu()
            output_features.append(features)
        
    
    # 转换为 numpy 数组
    input_features = torch.cat(input_features, dim=0).numpy()
    output_features = torch.cat(output_features, dim=0).numpy()
    

    # 计算 ResNet50 的 SPADE 分数
    spade_score_resnet50 = spade_score(input_features, output_features)
    print("ResNet50 的 SPADE 分数（鲁棒性指标）：", spade_score_resnet50)
