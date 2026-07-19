with open(r'D:\CODING\AETHYXLabs\AethyxLM\kaggle_train.ipynb', 'r', encoding='utf-8') as f:
    content = f.read()

idx = content.find('"cells"')
print(repr(content[idx:idx+500]))