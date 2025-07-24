import torch
import torch.nn.functional as F
from transformers import BertTokenizer
from model import TransformerModel

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载 tokenizer
tokenizer = BertTokenizer.from_pretrained(
    "C:/Users/24093/Desktop/LLM/bertbert_base_chinese",
    local_files_only=True
)

# 初始化并加载模型
model = TransformerModel(
    vocab_size=tokenizer.vocab_size,
    d_model=768,
    nhead=8,
    num_layers=36
)
checkpoint = torch.load(
    r"C:\Users\24093\Desktop\LLM\model\transformer_epoch0_step10000.pth",
    map_location=device
)
model.load_state_dict(checkpoint)
model.to(device).eval()

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """
    对 logits 做 Top-K 和/或 Top-P (nucleus) 过滤
    """
    assert logits.dim() == 1

    # Top-K
    if top_k > 0:
        topk_vals, topk_idx = torch.topk(logits, top_k)
        mask = logits < topk_vals[-1]
        logits[mask] = filter_value

    # Top-P
    if top_p > 0.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        # 找到 cumulative_probs > top_p 的 index
        cutoff = cumulative_probs > top_p
        # 第一个超过 p 的保留，之后都屏蔽
        cutoff_idx = torch.where(cutoff)[0]
        if cutoff_idx.numel() > 0:
            cutoff_pos = cutoff_idx[0].item()
            indices_to_remove = sorted_idx[cutoff_pos + 1 :]
            logits[indices_to_remove] = filter_value

    return logits

def causal_mask(size, device):
    """
    生成 causal mask，形状为 (size, size)，上三角填 -inf，下三角/对角线填 0
    """
    mask = torch.triu(torch.ones(size, size, device=device), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask

def chat(input_text, max_gen_len=50, temperature=1.0, top_k=50, top_p=0.9):
    # 编码
    input_ids = tokenizer.encode(input_text, add_special_tokens=True, return_tensors="pt").to(device)
    input_ids = input_ids[:, :128]  # 最大长度截断
    generated = input_ids

    for step in range(max_gen_len):
        seq_len = generated.size(1)
        mask = causal_mask(seq_len, device)

        with torch.no_grad():
            outputs = model(generated, mask)         # 假设 model 返回 (batch, seq_len, vocab_size)
        next_token_logits = outputs[0, -1, :]        # 取最后一个位置的 logits

        # 应用温度
        logits = next_token_logits / temperature

        # top_k top_p 过滤
        filtered_logits = top_k_top_p_filtering(logits.clone(), top_k=top_k, top_p=top_p)

        # 采样下一个 token
        probs = F.softmax(filtered_logits, dim=-1)
        next_token_id = torch.multinomial(probs, num_samples=1).unsqueeze(0)

        # 如果遇到 [SEP] 就停止
        if next_token_id.item() == tokenizer.sep_token_id:
            break

        # 拼接到序列末尾
        generated = torch.cat([generated, next_token_id], dim=1)

    # 解码生成部分
    output_ids = generated[0, input_ids.shape[1] :].cpu().tolist()
    response = tokenizer.decode(output_ids, skip_special_tokens=True)
    return response

if __name__ == "__main__":
    print("输入 “exit” 或 “quit” 退出")
    while True:
        user_input = input("用户：")
        if user_input.strip().lower() in ("exit", "quit"):
            break
        reply = chat(user_input)
        print("模型：", reply)
