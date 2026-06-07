# AASES — AI-Powered Answer Sheet Evaluation System

> Solo internship project | Deep Learning + OCR + NLP + Streamlit

## Quick Start

```bash
# 1. Clone & enter
git clone https://github.com/YOUR_USERNAME/AASES.git && cd AASES

# 2. Install as package (fixes all import issues)
pip install -e .
pip install -r requirements.txt

# 3. Copy env config
cp .env.example .env

# 4. Run OCR on your dataset
python -m src.ocr.ocr_ensemble \
  --csv data/sample/written_name_train_v2.csv \
  --images data/sample/train_v2 \
  --samples 50

# 5. Launch dashboard
streamlit run src/dashboard/app.py
```

## Dataset Structure Expected
```
data/
├── sample/
│   ├── train_v2/             ← handwritten name images
│   ├── test_v2/
│   ├── validation_v2/
│   ├── written_name_train_v2.csv      ← FILENAME, IDENTITY
│   ├── written_name_test_v2.csv
│   └── written_name_validation_v2.csv
```

## Modules Status
| Module | Status |
|---|---|
| M-01 Image Preprocessing | ✅ Complete |
| M-02 OCR Engine (Tesseract) | ✅ Complete |
| M-02 OCR Engine (TrOCR) | ⏳ Pending install |
| M-02 OCR Engine (PaddleOCR) | ⏳ Pending install |
| M-02 OCR Ensemble | ✅ Complete |
| M-03 Answer Key Manager | 🔲 Next |
| M-04 AI Evaluator Pass 1 | 🔲 Upcoming |
| M-05 AI Evaluator Pass 2 | 🔲 Upcoming |
| M-06 Conflict Resolver | 🔲 Upcoming |
| M-07 Teacher Dashboard | 🔲 Upcoming |
| M-08 Examiner Dashboard | 🔲 Upcoming |
