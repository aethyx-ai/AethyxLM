import re

with open('D:/CODING/AETHYXLabs/AethyxLM/colab_train.ipynb', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix the invalid escape sequence: \' inside double-quoted JSON string is invalid
# Replace the problematic pattern
fixed = content.replace(r"\\'{device_name}\\'", "'{device_name}'")

# Also check for any other similar issues
fixed = fixed.replace(r"\\'", "'")

with open('D:/CODING/AETHYXLabs/AethyxLM/colab_train.ipynb', 'w', encoding='utf-8') as f:
    f.write(fixed)

print('Fixed!')