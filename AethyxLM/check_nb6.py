with open(r'D:\CODING\AETHYXLabs\AethyxLM\kaggle_train.ipynb', 'r', encoding='utf-8') as f:
    import json
    nb = json.load(f)
    for i, cell in enumerate(nb['cells']):
        if i >= 10 and i <= 12:
            print(f'Cell {i}:')
            print(f'  type: {cell["cell_type"]}')
            print(f'  source lines: {len(cell["source"])}')
            for j, line in enumerate(cell['source']):
                if j < 3 or 'Warning' in line or 'GPU' in line or 'known_gpus' in line:
                    print(f'  line {j}: {repr(line)}')
            if 'metadata' in cell:
                print(f'  metadata: {cell["metadata"]}')
            print()