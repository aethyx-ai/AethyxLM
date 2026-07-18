with open('D:/CODING/AETHYXLabs/AethyxLM/colab_train.ipynb', 'r', encoding='utf-8') as f:
    content = f.read()

# The issue: literal backslash-singlequote in JSON string (\\')
# Replace the double-escaped backslash-singlequote with just single quote
fixed = content.replace('\\\\\'', "'")

with open('D:/CODING/AETHYXLabs/AethyxLM/colab_train.ipynb', 'w', encoding='utf-8') as f:
    f.write(fixed)

print('Fixed!')