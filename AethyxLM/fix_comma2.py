with open('D:/CODING/AETHYXLabs/AethyxLM/colab_train_production.ipynb', 'r', encoding='utf-8') as f:
    content = f.read()

# Find and fix the missing comma
# The line ends with ")" but should end with "),"
old = 'print(f"Warning: GPU \'{device_name}\' not in common Colab types {known_gpus}. Proceeding anyway...")\n   ]'
new = 'print(f"Warning: GPU \'{device_name}\' not in common Colab types {known_gpus}. Proceeding anyway..."),\n   ]'

content = content.replace(old, new)

with open('D:/CODING/AETHYXLabs/AethyxLM/colab_train_production.ipynb', 'w', encoding='utf-8') as f:
    f.write(content)

print('Fixed!')