import torch
from torch.utils.data import Dataset
import json

class DialogueDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=64):
        with open(data_path, 'r', encoding='utf-8') as f:
            self.dialogues = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        for dialogue in self.dialogues:
            # 拼接多轮对话为一条输入，最后一句为目标
            for i in range(1, len(dialogue)):
                src = " ".join(dialogue[:i])
                tgt = dialogue[i]
                self.samples.append((src, tgt))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        src, tgt = self.samples[idx]
        # 拼接输入和目标，目标右移一位作为label
        src_ids = self.tokenizer.encode(src, add_special_tokens=True, max_length=self.max_length, truncation=True)
        tgt_ids = self.tokenizer.encode(tgt, add_special_tokens=True, max_length=self.max_length, truncation=True)
        input_ids = src_ids + tgt_ids[:-1]
        labels = src_ids[1:] + tgt_ids
        input_ids = input_ids[:self.max_length]
        labels = labels[:self.max_length]
        # 补齐
        pad_id = self.tokenizer.pad_token_id or 0
        input_ids += [pad_id] * (self.max_length - len(input_ids))
        labels += [-100] * (self.max_length - len(labels))  # -100为ignore_index
        return torch.tensor(input_ids), torch.tensor(labels)