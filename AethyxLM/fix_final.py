with open('D:/CODING/AETHYXLabs/AethyxLM/colab_train_production.ipynb', 'r', encoding='utf-8') as f:
    content = f.read()

# The issue: missing comma after the print line in the JSON array
# Find the exact pattern and add comma
old = 'print(f"Warning: GPU \'{device_name}\' not in common Colab types {known_gpus}. Proceeding anyway...")\n   ]'
new = 'print(f"Warning: GPU \'{device_name}\' not in common Colab types {known_gpus}. Proceeding anyway..."),\n   ]'

content = content.replace(old, new)

with open('D:/CODING/AETHYXLabs/AethyxLM/colab_train_production.ipynb', 'w', encoding='utf-8') as f:
    f.write(content)

print('Fixed!')