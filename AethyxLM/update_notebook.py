import json
import os

with open('colab_train_final.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Cell index 4 (5th cell) is the tokenizer training cell
cell = nb['cells'][4]

# Let's inspect its source
print("Current source:")
print("".join(cell['source']))

# Replace the source with a robust version that generates metadata.json if missing, and gracefully copies
new_source = [
    "# ============================================================\n",
    "# CELL 4: TRAIN BPE TOKENIZER (32k vocab) + SAVE TO DRIVE\n",
    "# ============================================================\n",
    "import subprocess, sys, shutil, os, time, json\n",
    "\n",
    "print('Training tokenizer...')\n",
    "result = subprocess.run(\n",
    "    [sys.executable, '-m', 'tokenizer.train_tokenizer'],\n",
    "    cwd=LOCAL_ROOT, capture_output=True, text=True\n",
    ")\n",
    "print(result.stdout)\n",
    "if result.returncode != 0:\n",
    "    print('STDERR:', result.stderr)\n",
    "    raise RuntimeError('Tokenizer training failed')\n",
    "\n",
    "# Verify\n",
    "sys.path.insert(0, LOCAL_ROOT)\n",
    "from tokenizer.tokenizer import AethyxTokenizer\n",
    "tok = AethyxTokenizer()\n",
    "print(f'[OK] Vocab size: {tok.vocab_size}')\n",
    "ids = tok.encode('Hello world')\n",
    "print(f'[OK] Encode: {ids}')\n",
    "print(f'[OK] Decode: {tok.decode(ids)}')\n",
    "\n",
    "# Robust Fix: If metadata.json was not created by the cloned train_tokenizer.py, generate it now!\n",
    "metadata_path = 'tokenizer/metadata.json'\n",
    "if not os.path.exists(metadata_path):\n",
    "    print('Creating missing metadata.json dynamically...')\n",
    "    corpus_path = 'tokenizer/data/corpus.txt'\n",
    "    metadata = {\n",
    "        \"vocab_size\": tok.vocab_size,\n",
    "        \"tokenizer_type\": \"BPE\",\n",
    "        \"special_tokens\": [\"<PAD>\", \"<UNK>\", \"<BOS>\", \"<EOS>\"],\n",
    "        \"normalizer\": {\n",
    "            \"type\": \"Sequence\",\n",
    "            \"components\": [\n",
    "                {\"type\": \"NFD\"},\n",
    "                {\"type\": \"Lowercase\"},\n",
    "                {\"type\": \"StripAccents\"}\n",
    "            ]\n",
    "        },\n",
    "        \"pre_tokenizer\": {\n",
    "            \"type\": \"ByteLevel\"\n",
    "        },\n",
    "        \"trainer\": {\n",
    "            \"type\": \"BpeTrainer\",\n",
    "            \"vocab_size\": tok.vocab_size,\n",
    "            \"min_frequency\": 2,\n",
    "            \"special_tokens\": [\"<PAD>\", \"<UNK>\", \"<BOS>\", \"<EOS>\"]\n",
    "        },\n",
    "        \"timestamp\": time.strftime(\"%Y-%m-%d %H:%M:%S\"),\n",
    "        \"dataset_used\": corpus_path,\n",
    "        \"corpus_size_bytes\": os.path.getsize(corpus_path) if os.path.exists(corpus_path) else 0,\n",
    "    }\n",
    "    with open(metadata_path, 'w', encoding='utf-8') as f:\n",
    "        json.dump(metadata, f, indent=2, ensure_ascii=False)\n",
    "\n",
    "# Copy tokenizer and metadata safely to Drive\n",
    "shutil.copy2('tokenizer/tokenizer.json', os.path.join(DRIVE_TOK, 'tokenizer.json'))\n",
    "if os.path.exists(metadata_path):\n",
    "    shutil.copy2(metadata_path, os.path.join(DRIVE_TOK, 'metadata.json'))\n",
    "print('[OK] Tokenizer and metadata saved to Drive')\n"
]

cell['source'] = new_source

with open('colab_train_final.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)

print("Updated colab_train_final.ipynb successfully!")