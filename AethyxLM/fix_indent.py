with open('D:/CODING/AETHYXLabs/AethyxLM/scripts/prepare_fineweb.py', 'r') as f:
    content = f.read()

# Fix the _run_loop method definition indentation
content = content.replace(
    'def _run_loop(self):\n        """Main processing loop."""',
    '    def _run_loop(self):\n        """Main processing loop."""'
)

with open('D:/CODING/AETHYXLabs/AethyxLM/scripts/prepare_fineweb.py', 'w') as f:
    f.write(content)
print('Fixed')