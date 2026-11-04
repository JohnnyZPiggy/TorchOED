"""Execute all Python fences in order, in a fresh interpreter per page."""
from pathlib import Path
import re
import subprocess
import sys

for page in sorted(Path('website').glob('*.md')):
    blocks = re.findall(r'^```python\s*\n(.*?)^```\s*$', page.read_text(encoding='utf-8'), re.M | re.S)
    subprocess.run([sys.executable, '-c', '\n\n'.join(blocks)], check=True, timeout=60)
    print(f'{page}: {len(blocks)} Python blocks PASS', flush=True)
