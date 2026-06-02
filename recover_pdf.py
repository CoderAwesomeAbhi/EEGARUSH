"""Save full PDF text for recovery"""
import fitz

doc = fitz.open('paper/pdf/main.pdf')
full_text = ''
for page in doc:
    full_text += page.get_text()

with open('recovered_full.txt', 'w', encoding='utf-8') as f:
    f.write(full_text)
print(f'Saved {len(full_text)} chars')
