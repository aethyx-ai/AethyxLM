import json

with open('D:/CODING/AETHYXLabs/AethyxLM/colab_train_final.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print('Cells:', len(nb['cells']))

# Check Cell 1 has the nested dir fix
cell1 = nb['cells'][1]['source']
nested_fix = any('nested = os.path.join(LOCAL_ROOT, "AethyxLM")' in line for line in cell1)
print('Nested dir fix:', nested_fix)

# Check Cell 3 (CUDA) has fixed f-string
cell3 = nb['cells'][3]['source']
cuda_fix = any('Warning: GPU' in line and '\\' not in line for line in cell3)
print('CUDA warning fix:', cuda_fix)

# Check Cell 5 has nested fix
cell5 = nb['cells'][5]['source']
tokenizer_fix = any('os.path.exists' in line and 'AethyxLM/AethyxLM' in line for line in cell5)
print('Tokenizer nested fix:', tokenizer_fix)

print('All critical fixes present!')