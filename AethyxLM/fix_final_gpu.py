import json

with open('D:/CODING/AETHYXLabs/AethyxLM/colab_train_final.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Fix Cell 2 (index 2) - GPU warning line
cell = nb['cells'][2]
new_source = []
for line in cell['source']:
    if 'Warning: GPU' in line and '\\' in line:
        line = '    print(f"Warning: GPU \'{device_name}\' not in common Colab types {known_gpus}. Proceeding anyway...")'
    new_source.append(line)
cell['source'] = new_source

with open('D:/CODING/AETHYXLabs/AethyxLM/colab_train_final.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)

print("Fixed GPU warning line")