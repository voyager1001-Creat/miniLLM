import torch
import torch.nn as nn
import math
from transformers import BertModel, BertTokenizer

# Positional Encoding for Transformer models
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
    
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, nhead):
        super(MultiHeadAttention, self).__init__()
        assert d_model % nhead == 0
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)

    def scale_dot_product_attention(self, query, key, value, attn_mask=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attn_mask is not None:
            # attn_mask: (batch, seq_len, seq_len) or (seq_len, seq_len)
            # scores: (batch, nhead, seq_len, seq_len)
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            elif attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(1)
            scores = scores + attn_mask
        # 在softmax前，确保每一行至少有一个非-inf
        scores[scores != scores] = 0  # 先清理已有的nan
        mask_all_inf = torch.isinf(scores).all(dim=-1, keepdim=True)
        scores = scores.masked_fill(mask_all_inf, 0)
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = torch.where(torch.isnan(attn_weights), torch.zeros_like(attn_weights), attn_weights)
        context = torch.matmul(attn_weights, value)
        return context, attn_weights

    def forward(self, query, key, value, attn_mask=None):
        batch_size = query.size(0)
        seq_len = query.size(1)
        query = self.q_linear(query).view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        key = self.k_linear(key).view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        value = self.v_linear(value).view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        context, attn_weights = self.scale_dot_product_attention(query, key, value, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.out_linear(context)
    
class TransformerLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    @staticmethod
    def generate_causal_mask(seq_len):
        mask = torch.zeros(seq_len, seq_len)
        future = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        mask = mask.masked_fill(future, float("-inf"))
        return mask

    @staticmethod
    def generate_pad_mask(input_ids, pad_token_id=0):
        mask = (input_ids == pad_token_id)
        mask = mask.unsqueeze(1).unsqueeze(2)
        mask = mask.float().masked_fill(mask, float('-inf'))
        return mask

    def forward(self, src, causal_mask=None, pad_mask=None):
        batch_size, seq_len, _ = src.size()
        attn_mask = None
        if causal_mask is not None:
            attn_mask = causal_mask.unsqueeze(0).expand(batch_size, seq_len, seq_len)
        if pad_mask is not None:
            pad_mask_expanded = pad_mask.squeeze(1).expand(batch_size, seq_len, seq_len)
            if attn_mask is not None:
                attn_mask = attn_mask + pad_mask_expanded
            else:
                attn_mask = pad_mask_expanded
        src2 = self.self_attn(src, src, src, attn_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        feed_forward = self.feed_forward(src)
        src = src + self.dropout2(feed_forward)
        src = self.norm2(src)
        return src
    
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, dim_feedforward=2048, dropout=0.1, pad_token_id=0):
        super(TransformerModel, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.pad_token_id = pad_token_id
        self.layers = nn.ModuleList(
            [TransformerLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)])
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids, causal_mask=None, pad_mask=None):
        embeddings = self.embedding(input_ids) * math.sqrt(self.d_model)
        embeddings = self.pos_encoder(embeddings)
        if pad_mask is None:
            pad_mask = TransformerLayer.generate_pad_mask(input_ids, self.pad_token_id).to(embeddings.device)
        mask = causal_mask
        for layer in self.layers:
            embeddings = layer(embeddings, mask, pad_mask)
        return self.fc_out(embeddings)

if __name__ == "__main__":
    # 设置随机种子，保证可复现
    torch.manual_seed(42)
    import numpy as np
    np.random.seed(42)

    tokenizer = BertTokenizer.from_pretrained("C:\\Users\\24093\\Desktop\\LLM\\bertbert_base_chinese")
    bert_model = BertModel.from_pretrained("C:\\Users\\24093\\Desktop\\LLM\\bertbert_base_chinese")
    vocab_size = tokenizer.vocab_size
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    model = TransformerModel(
        vocab_size=tokenizer.vocab_size,
        d_model=768,
        nhead=8,
        num_layers=6,
        pad_token_id=pad_token_id
    )
    # 关键：模型参数初始化为相同值，避免未训练参数导致的差异
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.constant_(m.weight, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Embedding):
            nn.init.constant_(m.weight, 0.02)
    model.apply(init_weights)

    layer = TransformerLayer(d_model=8, nhead=2)
    causal_mask = layer.generate_causal_mask(5)
    print(causal_mask)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()  # 关键：关闭dropout，保证推理一致性

    L = 10
    K = 5
    A = torch.randint(0, vocab_size, (1, L)).to(device)
    B = torch.cat([A, torch.randint(0, vocab_size, (1, K)).to(device)], dim=1)
    padA = torch.full((1, L), pad_token_id, dtype=torch.long).to(device)
    padB = torch.cat([padA, torch.full((1, K), pad_token_id, dtype=torch.long).to(device)], dim=1)

    # 关键修正：为B构造一个只让前L个token能看到的causal mask
    causal_mask_B = torch.zeros(B.size(1), B.size(1), device=device)
    causal_mask_B[:L, :L] = TransformerLayer.generate_causal_mask(L)

    outA = model(A)[:, :L, :]
    outB = model(B, causal_mask=causal_mask_B)[:, :L, :]
    outPadA = model(padA)[:, :L, :]
    outPadB = model(padB, causal_mask=causal_mask_B)[:, :L, :]

    # 检查差异
    diff = (outA - outB).abs().max().item()
    print("最大差异:", diff)
    if not torch.allclose(outA, outB, atol=1e-5):
        print("警告：outA和outB的结果有较大差异，请检查模型实现或输入数据，必要时需要修改代码以保证两者输出一致。")
    assert torch.allclose(outA, outB, atol=1e-5)
    print("outPadA.shape:", outPadA.shape)
    print("outPadB.shape:", outPadB.shape)
    print("最大差异:", (outPadA - outPadB).abs().max().item())
    print("allclose:", torch.allclose(outPadA, outPadB, atol=1e-5))
    assert (outPadA - outPadB).abs().max().item() < 1e-5
    assert torch.allclose(outPadA, outPadB, atol=1e-6)
