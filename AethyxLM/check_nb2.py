import json

with open('kaggle_train.ipynb', 'r', encoding='utf-8') as f:
    content = f.read()

data = json.loads(content)
print(f'Total cells: {len(data["cells"])}')
for i, cell in enumerate(data['cells']):
    src = ''.join(cell['source'])
    if len(src) > 100:
        print(f'Cell {i}: {src[:80]}...')
    else:
        print(f'Cell {i}: {src}')