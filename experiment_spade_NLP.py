import torch
from transformers import BertTokenizer, BertModel
from torch.utils.data import DataLoader, Dataset
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import laplacian
import hnswlib
from datasets import load_dataset

# 自定义数据集类
class TextDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=128,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].squeeze()       # [seq_length]
        attention_mask = encoding['attention_mask'].squeeze()  # [seq_length]
        return input_ids, attention_mask

# hnsw 函数
def hnsw(features, k=10, ef=100, M=48):
    print("Initializing HNSW Index...")
    num_samples, dim = features.shape
    print(f"Number of samples: {num_samples}, Dimension: {dim}")
    p = hnswlib.Index(space='l2', dim=dim)
    p.init_index(max_elements=num_samples, ef_construction=ef, M=M)
    labels_index = np.arange(num_samples)
    p.add_items(features, labels_index)
    p.set_ef(ef)
    neighs, distances = p.knn_query(features, k + 1)
    print(f"Found nearest neighbors for k={k + 1}")
    return neighs, distances

# 构建邻接矩阵
def construct_adj(neighs, distances):
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

# 计算 SPADE 分数
def spade_score(input_features, output_features, k=10):
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

    # 计算特征值
    print("Computing eigenvalues...")
    eigvals_in = np.linalg.eigvalsh(L_in.toarray())
    eigvals_out = np.linalg.eigvalsh(L_out.toarray())

    # 计算 SPADE 分数
    spade_score_value = max(eigvals_out) / max(eigvals_in)
    print(f"SPADE Score: {spade_score_value}")
    return spade_score_value

if __name__ == "__main__":
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 加载预训练的 BERT 模型
    print("Loading pre-trained BERT model...")
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    bert_model.eval()
    bert_model.to(device)

    # 加载 IMDb 数据集
    print("Loading IMDb dataset...")
    subset = True
    if subset:
        dataset = load_dataset('imdb', split='test[:500]')  # 使用前 500 条数据
    else:
        dataset = load_dataset('imdb', split='test')
    texts = dataset['text']

    # 创建数据集和数据加载器
    text_dataset = TextDataset(texts)
    dataloader = DataLoader(text_dataset, batch_size=32, shuffle=False)

    input_features = []
    output_features = []

    print("Extracting input and output features...")
    with torch.no_grad():
        for input_ids, attention_mask in dataloader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            # 输入特征：词嵌入的平均值
            embeddings = bert_model.embeddings(input_ids)  # [batch_size, seq_length, hidden_size]
            avg_embeddings = embeddings.mean(dim=1).cpu().numpy()  # [batch_size, hidden_size]
            input_features.append(avg_embeddings)

            # 输出特征：模型最后一层的输出的平均值
            outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden_state = outputs.last_hidden_state  # [batch_size, seq_length, hidden_size]
            avg_output = last_hidden_state.mean(dim=1).cpu().numpy()  # [batch_size, hidden_size]
            output_features.append(avg_output)

    # 转换为 NumPy 数组
    input_features = np.concatenate(input_features, axis=0)
    output_features = np.concatenate(output_features, axis=0)

    # 计算 SPADE 分数
    spade_score_value = spade_score(input_features, output_features)
    print("BERT 模型的 SPADE 分数（鲁棒性指标）：", spade_score_value)
