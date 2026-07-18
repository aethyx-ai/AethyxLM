import json

with open('D:/CODING/AETHYXLabs/AethyxLM/colab_train.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    cell_type = cell['cell_type']
    source_len = len(cell['source'])
    print('Cell {}: {}, lines: {}'.format(i, cell_type, source_len))
    if cell['cell_type'] == 'code':
        for j, line in enumerate(cell['source'][:5]):
            print('  Line {}: {}'.format(j, repr(line[:80])))
    print()