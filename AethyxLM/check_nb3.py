import json

with open(r'D:\CODING\AETHYXLabs\AethyxLM\kaggle_train.ipynb', 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f'Total cells: {len(data["cells"])}')
for i, cell in enumerate(data['cells']):
    src = ''.join(cell['source'])
    if len(src) > 80:
        print(f'Cell {i}: {src[:80]}...')
    else:
        print(f'Cell {i}: {src}')