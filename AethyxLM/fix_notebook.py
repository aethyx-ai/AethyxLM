import json

with open('D:/CODING/AETHYXLabs/AethyxLM/colab_train.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Fix cell 2 (index 1) - the CUDA verification cell
cell = nb['cells'][1]

new_source = []
for line in cell['source']:
    # Fix the f-string with escaped single quotes
    if "GPU \\\\'" in line:
        line = line.replace("f'Warning: GPU \\\\'{device_name}\\\\',", "f\"Warning: GPU '{device_name}'")
    new_source.append(line)

cell['source'] = new_source

with open('D:/CODING/AETHYXLabs/AethyxLM/colab_train.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=2)

print('Fixed!')