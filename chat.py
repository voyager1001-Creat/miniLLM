import torch
from transformers import BertTokenizer
from model import TransformerModel

# 加载tokenizer和模型
tokenizer = BertTokenizer.from_pretrained("C:/Users/24093/Desktop/LLM/bertbert_base_chinese", local_files_only=True)
model = TransformerModel(
    vocab_size=tokenizer.vocab_size,
    d_model=768,
    nhead=8,
    num_layers=24
)
model.load_state_dict(torch.load("transformer_epoch9.pth", map_location="cpu"))  # 换成你最新的权重
model.eval()

def chat(input_text, max_gen_len=50):
    # 编码输入
    input_ids = tokenizer.encode(input_text, add_special_tokens=True, return_tensors="pt")
    input_ids = input_ids[:, :128]  # 截断
    generated = input_ids

    for _ in range(max_gen_len):
        mask = torch.triu(torch.ones(generated.shape[1], generated.shape[1]), diagonal=1).bool()
        with torch.no_grad():
            outputs = model(generated, mask)
        next_token_logits = outputs[0, -1, :]
        next_token_id = next_token_logits.argmax(-1).unsqueeze(0)
        # 如果生成了[SEP]就停止
        if next_token_id.item() == tokenizer.sep_token_id:
            break
        generated = torch.cat([generated, next_token_id.unsqueeze(0)], dim=1)

    response = tokenizer.decode(generated[0, input_ids.shape[1]:], skip_special_tokens=True)
    return response

if __name__ == "__main__":
    while True:
        user_input = input("你：")
        if user_input.strip().lower() in ["exit", "quit"]:
            break
        reply = chat(user_input)
        print("模型：", reply)