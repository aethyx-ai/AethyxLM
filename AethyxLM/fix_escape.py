with open('D:/CODING/AETHYXLabs/AethyxLM/colab_train_production.ipynb', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix the escaped single quotes in f-string - use double quotes for f-string instead
old = "print(f'Warning: GPU \\'{device_name}\\' not in common Colab types {known_gpus}. Proceeding anyway...')"
new = 'print(f"Warning: GPU \'{device_name}\' not in common Colab types {known_gpus}. Proceeding anyway...")'

content = content.replace(old, new)

with open('D:/CODING/AETHYXLabs/AethyxLM/colab_train_production.ipynb', 'w', encoding='utf-8') as f:
    f.write(content)

print('Fixed!')