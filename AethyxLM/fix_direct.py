with open('D:/CODING/AETHYXLabs/AethyxLM/colab_train_production.ipynb', 'r', encoding='utf-8') as f:
    content = f.read()

# Direct fix: find the exact position and insert comma
idx = content.find('Warning: GPU')
if idx >= 0:
    # Find the closing )" 
    end_idx = content.find(')\"\n   ]', idx)
    if end_idx >= 0:
        content = content[:end_idx+1] + ',' + content[end_idx+1:]
        print('Fixed at position', end_idx+1)
    else:
        print('Pattern not found')

with open('D:/CODING/AETHYXLabs/AethyxLM/colab_train_production.ipynb', 'w', encoding='utf-8') as f:
    f.write(content)

print('Done!')