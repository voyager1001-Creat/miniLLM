import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from model import TransformerModel
from dataset import DialogueDataset
import os
import glob

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained(
    "C:/Users/24093/Desktop/LLM/bertbert_base_chinese",
    local_files_only=True
)
dataset = DialogueDataset(r"C:/Users/24093/Downloads/LCCC-large/LCCD.json", tokenizer, max_length=128)
save_dir = r"C:/Users/24093/Desktop/LLM/model"
os.makedirs(save_dir, exist_ok=True)
saved_files = []
max_keep = 10

# 划分训练集和验证集
val_ratio = 0.05
val_size = int(len(dataset) * val_ratio)
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)

model = TransformerModel(
    vocab_size=tokenizer.vocab_size,
    d_model=768,
    nhead=8,
    num_layers=24
)
bert_model = BertModel.from_pretrained(
    "C:/Users/24093/Desktop/LLM/bertbert_base_chinese",
    local_files_only=True
)
model.embedding = nn.Embedding.from_pretrained(bert_model.embeddings.word_embeddings.weight, freeze=True)
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)

# 自动查找最新checkpoint
def find_latest_checkpoint():
    ckpt_files = glob.glob("transformer_epoch*.pth") + glob.glob("transformer_epoch*_step*.pth")
    ckpt_files += glob.glob(os.path.join(save_dir, "transformer_epoch*.pth"))
    if not ckpt_files:
        return None
    # 以修改时间排序，最新的在最后
    ckpt_files.sort(key=os.path.getmtime)
    return ckpt_files[-1]

start_epoch = 0
latest_ckpt = find_latest_checkpoint()
if latest_ckpt:
    print(f"恢复自 {latest_ckpt}")
    model.load_state_dict(torch.load(latest_ckpt, map_location=device))
    # 恢复epoch编号（假设文件名格式为 transformer_epoch{epoch}[_step{step}].pth）
    import re
    match = re.search(r'epoch(\d+)', latest_ckpt)
    if match:
        start_epoch = int(match.group(1)) + 1  # 下一个epoch开始

def evaluate(model, dataloader):
    model.eval()
    total_loss, total_correct, total_tokens = 0, 0, 0
    with torch.no_grad():
        for input_ids, labels in dataloader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            mask = torch.triu(torch.ones(input_ids.shape[1], input_ids.shape[1]), diagonal=1).bool().to(device)
            outputs = model(input_ids)
            loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
            total_loss += loss.item() * input_ids.size(0)
            preds = outputs.argmax(-1)
            mask_valid = labels != -100
            total_correct += ((preds == labels) & mask_valid).sum().item()
            total_tokens += mask_valid.sum().item()
    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = total_correct / total_tokens if total_tokens > 0 else 0
    return avg_loss, accuracy

for epoch in range(start_epoch, 10):
    model.train()
    total_loss, total_correct, total_tokens = 0, 0, 0
    for step, (input_ids, labels) in enumerate(train_loader, 1):
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        causal_mask = TransformerLayer.generate_causal_mask(input_ids.shape[1]).to(device)
        outputs = model(input_ids, causal_mask=causal_mask)
        loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        preds = outputs.argmax(-1)
        mask_valid = labels != -100
        total_correct += ((preds == labels) & mask_valid).sum().item()
        total_tokens += mask_valid.sum().item()
        if (step + 1) % 100 == 0:
            avg_loss = total_loss / (step)
            accuracy = total_correct / total_tokens if total_tokens > 0 else 0
            print(f"Epoch {epoch} Step {step+1}: Loss={avg_loss:.4f}, Accuracy={accuracy:.4f}")
        # 每100步保存模型
        if (step + 1) % 100 == 0:
            # 1. 保存新模型
            step_file = os.path.join(
                save_dir,
                f"transformer_epoch{epoch}_step{step}.pth"
            )
            torch.save(model.state_dict(), step_file)
            
            # 2. 更新列表并删除旧模型
            saved_files.append(step_file)
            if len(saved_files) > 10:
                oldest_file = saved_files.pop(0)  # 移除列表中最旧的文件
                if os.path.exists(oldest_file):
                    os.remove(oldest_file)
            
            print(f"Saved step checkpoint: {step_file}")

    avg_loss = total_loss / len(train_loader.dataset)
    accuracy = total_correct / total_tokens if total_tokens > 0 else 0
    print(f"Epoch {epoch} Train: Loss={avg_loss:.4f}, Accuracy={accuracy:.4f}")

    # 每个epoch评估
    val_loss, val_acc = evaluate(model, val_loader)
    print(f"Epoch {epoch} Val: Loss={val_loss:.4f}, Accuracy={val_acc:.4f}")

    # 保存模型
    torch.save(model.state_dict(),
           os.path.join(save_dir, f"transformer_epoch{epoch}.pth"))
