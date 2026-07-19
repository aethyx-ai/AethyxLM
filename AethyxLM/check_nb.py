import json
import nbformat
from nbformat.validator import validate

with open('kaggle_train.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f'Cells: {len(nb["cells"])}')
print('Cell types:', [c['cell_type'] for c in nb['cells']])

import nbformat
from nbformat.validator import validate
errors = validate(nb)
if errors:
    print('Validation errors:', errors)
else:
    print('Validation passed!')