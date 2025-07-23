import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from model import TransformerModel
from dataset import DialogueDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained("C:\\Users\\24093\\Desktop\\LLM\\bertbert_base_chinese")
dataset = DialogueDataset(r"C:\Users\24093\Downloads\LCCC-large\LCCD.json", tokenizer, max_length=128)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

model = TransformerModel(
    vocab_size=tokenizer.vocab_size,
    d_model=768,
    nhead=8,
    num_layers=24
)
bert_model = BertModel.from_pretrained("C:\\Users\\24093\\Desktop\\LLM\\bertbert_base_chinese")
model.embedding = nn.Embedding.from_pretrained(bert_model.embeddings.word_embeddings.weight, freeze=True)
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)

for epoch in range(10):
    model.train()
    for input_ids, labels in dataloader:
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        mask = torch.triu(torch.ones(input_ids.shape[1], input_ids.shape[1]), diagonal=1).bool().to(device)
        outputs = model(input_ids, mask)
        # outputs: [batch, seq, vocab]
        loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch} loss: {loss.item()}")
    # 保存模型参数
    torch.save(model.state_dict(), f"transformer_epoch{epoch}.pth")