import torch
from model.gpt import GPT
from tokenizer.tokenizer import AethyxTokenizer
import sys

# Quick test
model = GPT()
model.eval()

from tokenizer.tokenizer import AethyxTokenizer
tok = AethyxTokenizer()
print("Model and tokenizer loaded successfully")

# Quick generation test
@torch.no_grad()
def gen(prompt, max_new=30):
    ids = torch.tensor([tok.encode(prompt)], dtype=torch.long)
    for _ in range(15):
        logits = model(ids[:, -128:])
        logits = logits[:, -1, :] / 0.8
        v, _ = torch.topk(logits, 50)
        logits[logits < v[:, [-1]]] = -float('inf')
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, 1)
        ids = torch.cat([ids, next_id], dim=1)
    return tok.decode(ids[0].tolist())

# Load model
ckpt = torch.load('checkpoints/checkpoint_step_3090.pt', map_location='cpu', weights_only=False)
model = GPT()
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

from tokenizer.tokenizer import AethyxTokenizer
tok = AethyxTokenizer()

print('Test 1:', gen('Once upon a time')[:80])
print('Test 2:', gen('The little boy')[:80])
print('Test 3:', gen('Hello')[:50])
print('All inference tests passed!')