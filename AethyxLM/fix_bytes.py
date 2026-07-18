with open('D:/CODING/AETHYXLabs/AethyxLM/colab_train.ipynb', 'rb') as f:
    content = f.read()

# Fix the invalid escape: \\\\' inside f-string should be just '
# The bad sequence is: f'Warning: GPU \\'{device_name}\\',
# In bytes it appears as: f'Warning: GPU \\'{device_name}\',
# The JSON shows: f'Warning: GPU \\'{device_name}\\',
# So we need to replace the literal backslash-quote with just quote

bad = b"f'Warning: GPU \\'{device_name}\\',"
good = b'f"Warning: GPU \'{device_name}\',"'

content = content.replace(bad, good)

with open('D:/CODING/AETHYXLabs/AethyxLM/colab_train.ipynb', 'wb') as f:
    f.write(content)

print('Fixed!')