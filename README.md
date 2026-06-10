# Support Integrity Auditor (SIA)

AI-powered CRM ticket priority mismatch detector using self-supervised learning and fine-tuned DistilBERT.

## Live App
https://support-integrity-auditor-jmjj2rz2pkegqjpkaof99h.streamlit.app

## Results

| Metric | Score | Threshold | Status |
|--------|-------|-----------|--------|
| Accuracy | 87.42% | 83% | Pass |
| Macro F1 | 0.854 | 0.82 | Pass |
| Recall Consistent | 0.877 | 0.78 | Pass |
| Recall Mismatch | 0.868 | 0.78 | Pass |

## Pipeline

**Stage 1 - Pseudo Label Generation**
- Signal 1: Rule-based NLP keyword scoring (weight 0.3)
- Signal 2: Resolution time normalization (weight 0.2)
- Signal 3: Semantic embeddings via all-MiniLM-L6-v2 (weight 0.5)
- Mismatch threshold: delta > 0.30
- Mismatch rate: 28.9% (5,782 of 20,000 tickets)

**Stage 2 - Classifier Training**
- Model: distilbert-base-uncased fine-tuned
- Input: text + channel + priority metadata
- Imbalance: WeightedRandomSampler
- Epochs: 4, lr=2e-5, batch_size=32

**Stage 3 - Evidence Dossier**
- mismatch_type: Hidden Crisis or False Alarm
- feature_evidence: grounded to input fields
- Zero hallucination policy enforced

## Ablation Study

| Signal | Weight | Critical Score | Low Score | Separation |
|--------|--------|---------------|-----------|------------|
| Rule NLP | 0.3 | 0.556 | 0.472 | 0.084 |
| Resolution Time | 0.2 | 0.236 | 0.552 | -0.317 |
| Semantic Embedding | 0.5 | 0.760 | 0.423 | 0.337 |
| All Combined | - | 0.594 | 0.464 | 0.130 |

Semantic embedding has the highest separation (0.337), justifying its 0.5 weight.

## Dataset
Customer Support Tickets - CRM Dataset (Kaggle, 20,000 tickets)

## Model
HuggingFace: arshit0101/support-integrity-auditor

## Files
- app.py - Streamlit web app
- requirements.txt - Dependencies
- predictions.csv - Test set predictions
- dossiers.json - Evidence dossiers
- full_results.csv - Full dataset with signals
- ablation_table.csv - Ablation results

## Install
pip install -r requirements.txt
streamlit run app.py
