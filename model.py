import torch
import torch.nn as nn
import math
from transformers import BertModel, BertTokenizer

# Positional Encoding for Transformer models
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        # 正弦位置编码。d_model:输入的特征维度 max_len:最大序列长度
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 生成位置编码矩阵
        pe = torch.zeros(max_len, d_model) # 形状（max_len, d_model）
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term) # 正弦编码在偶数位置
        pe[:, 1::2] = torch.cos(position * div_term) # 余弦编码在奇数位置

        pe = pe.unsqueeze(0)  # 添加批次维度，现状（1, max_len, d_model）
        self.register_buffer('pe', pe)

    # 前向传播函数。 x:输入的特征矩阵，形状（batch_size, seq_len, d_model）
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        # returns 添加位置编码后的张量
        return self.dropout(x)
    
# 多头注意力机制
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, nhead):  # d_model:特征维度 nhead:头数
        super(MultiHeadAttention, self).__init__()
        assert d_model % nhead == 0  # d_model 必须被 nhead 整除
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead # 每个头的维数

        # 定义线性变换层
        self.q_linear = nn.Linear(d_model, d_model) # 定义查询变换
        self.k_linear = nn.Linear(d_model, d_model) # 定义键变换
        self.v_linear = nn.Linear(d_model, d_model) # 定义值变换
        self.out_linear = nn.Linear(d_model, d_model) # 定义输出变换

    def scale_dot_product_attention(self, query, key, value, mask=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim) # 缩放点积注意力
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = torch.softmax(scores, dim=-1) # softmax归一化
        context = torch.matmul(attn_weights, value) # 加权求和，计算上下文向量
        return context,attn_weights  # 返回上下文向量和注意力权重

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 线性变换
        query = self.q_linear(query).view(batch_size, -1, self.nhead, self.head_dim).transpose(1, 2)
        key = self.k_linear(key).view(batch_size, -1, self.nhead, self.head_dim).transpose(1, 2)
        value = self.v_linear(value).view(batch_size, -1, self.nhead, self.head_dim).transpose(1, 2)

        context, attn_weights = self.scale_dot_product_attention(query, key, value, mask) # 调用scale_dot_product_attention函数计算上下文和注意力权重
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model) # 合并头

        return self.out_linear(context)  # 返回形状为 (batch_size, seq_len, d_model) 的张量
    
#Transformer层
class TransformerLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1): # 定义单个Transfomer层。d_model:特征维度 nhead:头数 dim_feedforward:前馈网络的维度 dropout:dropout比率
        super(TransformerLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead)  # 自注意力机制
        self.norm1 = nn.LayerNorm(d_model)  # 层归一化
        self.dropout1 = nn.Dropout(dropout)  # dropout层

        #前馈子层
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),  # 前馈网络的第一层
            nn.ReLU(),  # 激活函数
            nn.Linear(dim_feedforward, d_model)  # 前馈网络的第二层
        )
        self.norm2 = nn.LayerNorm(d_model)  # 层归一化
        self.dropout2 = nn.Dropout(dropout)  # dropout层

    # 生成因果掩码（防止看到未来信息）
    def generate_causal_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, src_mask=None):
        if src_mask is None:
            src_mask = self.generate_causal_mask(src.size(1)).to(src.device)
        src2 = self.self_attn(src, src, src, src_mask)  # 自注意力机制
        src = src + self.dropout1(src2)  # 残差连接
        src = self.norm1(src)  # 归一化

        feed_forward = self.feed_forward(src)  # 前馈网络
        src = src + self.dropout2(feed_forward)  # 残差连接
        src = self.norm2(src)  # 归一化

        return src  # 返回形状为 (batch_size, seq_len, d_model) 的张量
    
#完整Transformer模型
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, dim_feedforward=2048, dropout=0.1): # 定义完整的Transformer模型。d_model:特征维度 nhead:头数 num_layers:Transformer层数 dim_feedforward:前馈网络的维度 dropout:dropout比率
        super(TransformerModel, self).__init__()
        self.d_model = d_model  # 保存d_model以便后续使用
        self.embedding = nn.Embedding(vocab_size, d_model)  # 词嵌入层
        self.pos_encoder = PositionalEncoding(d_model, dropout)  # 位置编码

        self.layers = nn.ModuleList(
            [TransformerLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)])  # 多个Transformer层
        self.fc_out = nn.Linear(d_model, vocab_size)  # 输出层

    def forward(self, input_ids, src_mask=None):
        # 获取词嵌入
        embeddings = self.embedding(input_ids) * math.sqrt(self.d_model)
        embeddings = self.pos_encoder(embeddings)  # 添加位置编码

        mask = src_mask
        for layer in self.layers:
            embeddings = layer(embeddings, mask)  # 通过每个Transformer层
        return self.fc_out(embeddings)  # 输出词表概率

#训练辅助函数
def generate_mask(len):
    # 生成自注意力掩码
    mask = torch.triu(torch.ones(len, len), diagonal=1).bool()  # 上三角矩阵
    return mask  # 返回形状为 (len, len) 的布尔掩码矩阵

if __name__ == "__main__":


#模型实例
    tokenizer = BertTokenizer.from_pretrained("C:\\Users\\24093\\Desktop\\LLM\\bertbert_base_chinese")
    model = TransformerModel(
    vocab_size=tokenizer.vocab_size,  # 需传入词表大小
    d_model=768,
    nhead=8,
    num_layers=6
)
    input_text = "你好，世界！"  # 输入文本
    # 用BERT权重初始化嵌入层
    bert_model = BertModel.from_pretrained("C:\\Users\\24093\\Desktop\\LLM\\bertbert_base_chinese")
    input_ids = tokenizer.encode(input_text, return_tensors='pt')  # [1, seq_len]
    model.embedding = nn.Embedding.from_pretrained(bert_model.embeddings.word_embeddings.weight, freeze=True)  # 冻结嵌入层

# 测试模型
    mask = generate_mask(input_ids.shape[1])  # 根据实际输入长度生成掩码
    output = model(input_ids, mask)  # 前向传播
    print(output.shape)  # 输出形状应为 (1, 20, vocab_size)
