import torch
from model import TransformerModel
from transformers import BertTokenizer, BertModel
import torch.nn as nn

tokenizer = BertTokenizer.from_pretrained("C:\\Users\\24093\\Desktop\\LLM\\bertbert_base_chinese")
model = TransformerModel(
    vocab_size=tokenizer.vocab_size,
    d_model=768,
    nhead=8,
    num_layers=24
)
bert_model = BertModel.from_pretrained("C:\\Users\\24093\\Desktop\\LLM\\bertbert_base_chinese")
model.embedding = nn.Embedding.from_pretrained(bert_model.embeddings.word_embeddings.weight, freeze=True)
model.load_state_dict(torch.load(r"C:\Users\24093\Desktop\LLM\miniLLM.pth", map_location="cpu"))
model.eval()

dummy_input = torch.ones(1, 64, dtype=torch.long)
mask = torch.triu(torch.ones(64, 64), diagonal=1).bool()
torch.onnx.export(
    model,
    (dummy_input, mask),
    "transformer_llm.onnx",
    input_names=["input_ids", "mask"],
    output_names=["output"],
    dynamic_axes={"input_ids": {1: "seq_len"}, "mask": {0: "seq_len", 1: "seq_len"}, "output": {1: "seq_len"}},
    opset_version=13
)