import json

with open('D:/CODING/AETHYXLabs/AethyxLM/colab_train_final.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print("Loaded notebook with", len(nb['cells']), "cells")

# Fix 1: Cell 2 (index 1) - CUDA warning f-string - fix the escaped single quotes
cell2 = nb['cells'][2]
new_source2 = []
for line in cell2['source']:
    if 'Warning: GPU' in line and '\\\\' in line:
        # Replace the problematic line with a clean version using double quotes for f-string
        line = '    print(f"Warning: GPU \'{device_name}\' not in common Colab types {known_gpus}. Proceeding anyway...")'
    new_source2.append(line)
cell2['source'] = new_source2

# Fix 2: Cell 1 (index 1) - Add nested directory fix after os.chdir
cell1 = nb['cells'][1]
new_source1 = []
for line in cell1['source']:
    new_source1.append(line)
    if 'os.chdir(LOCAL_ROOT)' in line:
        # Add nested directory fix after chdir
        new_source1.append('\n')
        new_source1.append('# Fix nested directory from git clone (repo clones to /content/AethyxLM/AethyxLM)\n')
        new_source1.append('nested = os.path.join(LOCAL_ROOT, \'AethyxLM\')\n')
        new_source1.append('if os.path.exists(nested):\n')
        new_source1.append('    for item in os.listdir(nested):\n')
        new_source1.append('        src = os.path.join(nested, item)\n')
        new_source1.append('        dst = os.path.join(LOCAL_ROOT, item)\n')
        new_source1.append('        if os.path.exists(dst):\n')
        new_source1.append('            if os.path.isdir(dst):\n')
        new_source1.append('                shutil.rmtree(dst)\n')
        new_source1.append('            else:\n')
        new_source1.append('                os.remove(dst)\n')
        new_source1.append('        shutil.move(src, LOCAL_ROOT)\n')
        new_source1.append('    os.rmdir(nested)\n')
        new_source1.append('\n')
cell1['source'] = new_source1

# Fix 3: Cell 4 (tokenizer) - verify it has the metadata fix, if not add it
cell4 = nb['cells'][5]
source_text = ''.join(cell4['source'])
if 'metadata_path' not in ''.join(cell4['source']):
    # Add the metadata generation and safe copy
    new_source4 = []
    for line in cell4['source']:
        new_source4.append(line)
        if 'print(\'[OK] Tokenizer saved to Drive\')' in line:
            # Add metadata generation before the final print
            new_source4.insert(-1, '\n')
            new_source4.insert(-1, '# Robust Fix: If metadata.json was not created by the cloned train_tokenizer.py, generate it now!\n')
            new_source4.insert(-1, 'metadata_path = \'tokenizer/metadata.json\'\n')
            new_source4.insert(-1, 'if not os.path.exists(metadata_path):\n')
            new_source4.insert(-1, '    print(\'Creating missing metadata.json dynamically...\')\n')
            new_source4.insert(-1, '    corpus_path = \'tokenizer/data/corpus.txt\'\n')
            new_source4.insert(-1, '    metadata = {\n')
            new_source4.insert(-1, '        "vocab_size": tok.vocab_size,\n')
            new_source4.insert(-1, '        "tokenizer_type": "BPE",\n')
            new_source4.insert(-1, '        "special_tokens": ["<PAD>", "<UNK>", "<BOS>", "<EOS>"],\n')
            new_source4.insert(-1, '        "normalizer": {\n')
            new_source4.insert(-1, '            "type": "Sequence",\n')
            new_source4.insert(-1, '            "components": [\n')
            new_source4.insert(-1, '                {"type": "NFD"},\n')
            new_source4.insert(-1, '                {"type": "Lowercase"},\n')
            new_source4.insert(-1, '                {"type": "StripAccents"}\n')
            new_source4.insert(-1, '            ]\n')
            new_source4.insert(-1, '        },\n')
            new_source4.insert(-1, '        "pre_tokenizer": {\n')
            new_source4.insert(-1, '            "type": "ByteLevel"\n')
            new_source4.insert(-1, '        },\n')
            new_source4.insert(-1, '        "trainer": {\n')
            new_source4.insert(-1, '            "type": "BpeTrainer",\n')
            new_source4.insert(-1, '            "vocab_size": tok.vocab_size,\n')
            new_source4.insert(-1, '            "min_frequency": 2,\n')
            new_source4.insert(-1, '            "special_tokens": ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]\n')
            new_source4.insert(-1, '        },\n')
            new_source4.insert(-1, '        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),\n')
            new_source4.insert(-1, '        "dataset_used": corpus_path,\n')
            new_source4.insert(-1, '        "corpus_size_bytes": os.path.getsize(corpus_path) if os.path.exists(corpus_path) else 0,\n')
            new_source4.insert(-1, '    }\n')
            new_source4.insert(-1, '    with open(metadata_path, \'w\', encoding=\'utf-8\') as f:\n')
            new_source4.insert(-1, '        json.dump(metadata, f, indent=2, ensure_ascii=False)\n')
            new_source4.insert(-1, '\n')
            new_source4.insert(-1, '# Copy tokenizer and metadata safely to Drive\n')
            new_source4.insert(-1, "shutil.copy2('tokenizer/tokenizer.json', os.path.join(DRIVE_TOK, 'tokenizer.json'))\n")
            new_source4.insert(-1, "if os.path.exists(metadata_path):\n")
            new_source4.insert(-1, "    shutil.copy2(metadata_path, os.path.join(DRIVE_TOK, 'metadata.json'))\n")
            new_source4.insert(-1, "print('[OK] Tokenizer and metadata saved to Drive')\n")
            new_source4.insert(-1, '\n')
            # Remove the old copy lines
            new_source4 = [l for l in new_source4 if 'shutil.copy2' not in l or 'metadata.json' not in l]
            cell4['source'] = new_source4

# Write the fixed notebook
with open('D:/CODING/AETHYXLabs/AethyxLM/colab_train_final.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)

print("All fixes applied! Verifying JSON...")
import json
with open('D:/CODING/AETHYXLabs/AethyxLM/colab_train_final.ipynb', 'r', encoding='utf-8') as f:
    json.load(f)
print("✅ Valid JSON!")
print("✅ All fixes applied!")