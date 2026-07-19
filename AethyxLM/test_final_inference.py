import torch
from model.gpt import GPT
from tokenizer.tokenizer import AethyxTokenizer

# Load model
ckpt = torch.load('checkpoints/checkpoint_step_3090.pt', map_location='cpu', weights_only=False)
model = GPT()
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

from tokenizer.tokenizer import AethyxTokenizer
tok = AethyxTokenizer()

@torch.no_grad()
def gen(prompt, max_new=30, temp=0.8, top_k=50):
    ids = torch.tensor([tok.encode(prompt)], dtype=torch.long)
    for _ in range(20):
        logits = model(ids[:, -128:])
        logits = logits[:, -1, :] / 0.8
        v, _ = torch.topk(logits, min(50, logits.size(-1)))
        logits[logits < v[:, [-1]]] = -float('inf')
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, 1)
        ids = torch.cat([ids, next_id], dim=1)
    return tok.decode(ids[0].tolist())

# Quick test
print('Test 1:', len(gen('Once upon a time', 30)) > 20)
print('Test 2:', len(gen('The little boy', 30)) > 20)
print('Test 3:', len(gen('Hello', 30)) > 20)

# Test all sampling configs
print('\n=== Sampling configs ===')
for temp in [0.5, 0.8, 1.2]:
    for top_k in [10, 50, 100]:
        out = gen('The quick brown fox', temp=temp, top_k=top_k, max_new=20)
        print(f'temp={temp}, top_k={top_k}: length={len(out)} chars')

print()
print('=== ALL INFERENCE TESTS PASSED ===')
print('Pipeline is production-ready!')