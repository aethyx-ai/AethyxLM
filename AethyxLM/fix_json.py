with open('D:/CODING/AETHYXLabs/AethyxLM/colab_train.ipynb', 'rb') as f:
    content = f.read()

# The issue: in the JSON text there's a literal backslash-quote sequence \'
# which is an invalid JSON escape. We need to replace backslash-quote with just quote.
# In the raw file, this appears as two bytes: backslash (0x5c) followed by quote (0x27)

# Count occurrences
count = content.count(b'\\\'')
print(f'Found {count} occurrences of backslash-quote')

# Replace backslash-quote with just quote
fixed = content.replace(b"\\'", b"'")

with open('D:/CODING/AETHYXLabs/AethyxLM/colab_train.ipynb', 'wb') as f:
    f.write(fixed)

print('Fixed!')