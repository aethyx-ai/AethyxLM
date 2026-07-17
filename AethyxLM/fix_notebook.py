with open('D:/CODING/AETHYXLabs/AethyxLM/colab_train.ipynb', 'r') as f:
    content = f.read()
content = content.replace("PROJECT_ROOT = '/content'", "PROJECT_ROOT = '/content/AethyxLM'")
content = content.replace("Zip extracts to /content/ (configs/, tokenizer/, model/, etc. directly)", "Zip extracts to /content/AethyxLM/ (configs/, tokenizer/, model/, etc.)")
with open('D:/CODING/AETHYXLabs/AethyxLM/colab_train.ipynb', 'w') as f:
    f.write(content)
print('Fixed')