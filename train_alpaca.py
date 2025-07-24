import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from model import TransformerModel
from alpaca_dataset import AlpacaDataset  # 使用新的Dataset
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained(
    "C:/Users/24093/Desktop/LLM/bertbert_base_chinese",
    local_files_only=True
)
dataset = AlpacaDataset("C:/Users/24093/Downloads/alpaca_data.json", tokenizer, max_length=128)
saved_files = ["C:/Users/24093/Desktop/LLM/model"]

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

# 如果继续训练，加载已有权重
if torch.cuda.is_available():
    model.load_state_dict(torch.load("transformer_final.pth"))
else:
    model.load_state_dict(torch.load("transformer_final.pth", map_location="cpu"))

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)

def evaluate(model, dataloader):
    model.eval()
    total_loss, total_correct, total_tokens = 0, 0, 0
    with torch.no_grad():
        for input_ids, labels in dataloader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            mask = torch.triu(torch.ones(input_ids.shape[1], input_ids.shape[1]), diagonal=1).bool().to(device)
            outputs = model(input_ids, mask)
            loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
            total_loss += loss.item() * input_ids.size(0)
            preds = outputs.argmax(-1)
            mask_valid = labels != -100
            total_correct += ((preds == labels) & mask_valid).sum().item()
            total_tokens += mask_valid.sum().item()
    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = total_correct / total_tokens if total_tokens > 0 else 0
    return avg_loss, accuracy

for epoch in range(5):
    model.train()
    total_loss, total_correct, total_tokens = 0, 0, 0
    for step, (input_ids, labels) in enumerate(train_loader):
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        mask = torch.triu(torch.ones(input_ids.shape[1], input_ids.shape[1]), diagonal=1).bool().to(device)
        outputs = model(input_ids, mask)
        loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * input_ids.size(0)
        preds = outputs.argmax(-1)
        mask_valid = labels != -100
        total_correct += ((preds == labels) & mask_valid).sum().item()
        total_tokens += mask_valid.sum().item()
        if (step + 1) % 100 == 0:
            avg_loss = total_loss / ((step + 1) * input_ids.size(0))
            accuracy = total_correct / total_tokens if total_tokens > 0 else 0
            print(f"Epoch {epoch} Step {step+1}: Loss={avg_loss:.4f}, Accuracy={accuracy:.4f}")
        # 每100步保存模型
        if (step + 1) % 100 == 0:
            # 1. 保存新模型
            step_file = f"transformer_epoch{epoch}_step{step+1}.pth"
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

    val_loss, val_acc = evaluate(model, val_loader)
    print(f"Epoch {epoch} Val: Loss={val_loss:.4f}, Accuracy={val_acc:.4f}")

    torch.save(model.state_dict(), f"transformer_alpaca_epoch{epoch}.pth")