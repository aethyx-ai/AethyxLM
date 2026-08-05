import sys
import json
import torch
sys.path.insert(0, 'D:/CODING/AETHYXLabs/AethyxLM')
from model.gpt import GPT

with open('D:/CODING/AETHYXLabs/AethyxLM/configs/train_config_modern.json', 'r') as f:
    full_config = json.load(f)

config = full_config['model']
model = GPT(config=config)
print('Modern architecture loaded successfully!')
print('Model params:', sum(p.numel() for p in model.parameters()))

# Test forward pass
x = torch.randint(0, 100, (2, 64))
with torch.no_grad():
    out = model(x)
    print('Output shape:', out.shape)