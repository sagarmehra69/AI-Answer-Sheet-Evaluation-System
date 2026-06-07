import json

with open('data/processed/ocr_results.json') as f:
    data = json.load(f)

for r in data[:5]:
    print('File    :', r['filename'])
    print('Expected:', r['ground_truth'])
    print('Got     :', r['tesseract_text'])
    print('Conf    :', round(r['tesseract_conf'], 3))
    print()