import torch
from torch.utils.data import Dataset
import json

class AlpacaDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=128):
        with open(data_path, 'r', encoding='utf-8') as f:
            self.samples = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        # 拼接prompt
        if item.get("input", "").strip():
            prompt = f"{item['instruction']}\n{item['input']}"
        else:
            prompt = item['instruction']
        target = item['output']

        # 编码
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=True, max_length=self.max_length, truncation=True)
        target_ids = self.tokenizer.encode(target, add_special_tokens=True, max_length=self.max_length, truncation=True)

        # 拼接输入和标签（SFT常用方式：输入+输出，label为输入部分为-100，输出部分为真实token）
        input_ids = prompt_ids + target_ids
        input_ids = input_ids[:self.max_length]
        labels = [-100] * len(prompt_ids) + target_ids
        labels = labels[:self.max_length]

        # 补齐
        pad_id = self.tokenizer.pad_token_id or 0
        input_ids += [pad_id] * (self.max_length - len(input_ids))
        labels += [-100] * (self.max_length - len(labels))

        return torch.tensor(input_ids), torch.tensor(labels)