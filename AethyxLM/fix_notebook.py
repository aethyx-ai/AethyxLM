import json

with open('D:/CODING/AETHYXLabs/AethyxLM/kaggle_train.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Fix: ensure all cells have proper structure
for i, cell in enumerate(nb['cells']):
    if 'metadata' not in cell:
        cell['metadata'] = {}
    if 'outputs' not in cell:
        cell['outputs'] = []
    if 'execution_count' not in cell:
        cell['execution_count'] = None

with open('kaggle_train_fixed.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)

print('Fixed and saved as kaggle_train_fixed.ipynb')