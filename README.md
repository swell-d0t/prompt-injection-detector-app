# Prompt Injection Detector: Group 5C (AI4ALL Ignite)

**Live app:** https://prompt-injection-detector-app.streamlit.app/

## Overview

This project evaluates and compares machine learning approaches for detecting prompt injection attacks; malicious inputs designed to hijack or manipulate large language models. We fine-tuned a DistilBERT transformer model and benchmarked it against three classical ML baselines (Logistic Regression, SVM, and Random Forest), with a specific focus on **adversarial robustness**: how well each model holds up against attacks designed to evade detection, not just how well it performs on clean, unperturbed data.

## What's in this repo

- `app.py` — Streamlit app with two views:
  - **Try the Detector:** test any prompt against DistilBERT or any of the classical baselines
  - **Robustness Dashboard:** visualizes detection rates across 20 different adversarial evasion techniques
- `requirements.txt` — dependencies
- `models/` — trained classical ML models (Logistic Regression, Random Forest, SVM) and their matching vectorizers/embedders
- `mindgard_results.csv` — adversarial robustness results used in the dashboard

The DistilBERT model itself is hosted separately on Hugging Face: [swell-d0t/prompt-injection-distilbert-v2](https://huggingface.co/swell-d0t/prompt-injection-distilbert-v2)

## Models

| Model | Type | Notes |
|---|---|---|
| DistilBERT (v2) | Transformer | Fine-tuned on xTRam1/safe-guard-prompt-injection + Tensor Trust (synthetic and human attacks) |
| Logistic Regression (TF-IDF) | Classical ML | TF-IDF features |
| Logistic Regression (Embeddings) | Classical ML | Sentence embeddings via all-MiniLM-L6-v2 |
| SVM | Classical ML | Linear kernel, TF-IDF features |
| Random Forest | Classical ML | TF-IDF features, tuned via GridSearchCV |

## Key findings

Initial testing showed DistilBERT scoring near-perfect accuracy on its training distribution, but with a major blind spot: it failed to detect prompt injection attacks using Unicode manipulation (homoglyphs, invisible characters, unusual spacing), evading detection almost entirely. Retraining on a more diverse dataset (Tensor Trust, made up of real human-submitted adversarial attacks rather than synthetic data) closed most of this gap without any additional preprocessing. This suggests the original vulnerability was substantially a training-data diversity problem rather than a fixed architectural limitation.

## Team

Ariana, Benchy, Simon, Francis: AI4ALL Ignite Accelerator-- Group 5C

## Citations

Liu, Y., et al. (2023, updated 2025). Prompt Injection Attack Against LLM-Integrated Applications. arXiv:2306.05499. https://arxiv.org/abs/2306.05499
OWASP. (2025). OWASP Top 10 for LLM Applications 2025. https://genai.owasp.org/resource/owasp-top-10-for-llm-applications-2025/ 
Morris, J., et al. (2020). TextAttack: A Framework for Adversarial Attacks, Data Augmentation, and Adversarial Training in NLP. arXiv:2005.05909. https://arxiv.org/abs/2005.05909 
Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why Should I Trust You?": Explaining the Predictions of Any Classifier. arXiv:1602.04938. https://arxiv.org/abs/1602.04938 
Pérez, F., & Ribeiro, I. (2022). Ignore Previous Prompt: Attack Techniques for Language Models. arXiv:2211.09527. https://arxiv.org/abs/2211.09527 
HackAPrompt: Exposing Systemic Vulnerabilities of LLMs through a Global Scale Prompt Hacking Competition. Schulhoff, S., et al. (2023). Published at EMNLP 2023.https://arxiv.org/abs/2311.16119
Toyer, S., Watkins, O., Mendes, E. A., Svegliato, J., Bailey, L., Wang, T., Ong, I., Elmaaroufi, K., Abbeel, P., Darrell, T., Ritter, A., & Russell, S. (2024). Tensor Trust: Interpretable Prompt Injection Attacks from an Online Game. International Conference on Learning Representations (ICLR). arXiv:2311.01011.

## Next steps

### Technical extensions:

Fold Random Forest and both LogReg variants into the same Mindgard/HackAPrompt robustness testing pipeline DistilBERT went through. This would display a full head-to-head comparison across all 5 models on adversarial robustness, not just clean accuracy
Explore whether data augmentation (training directly on Unicode-obfuscated examples, rather than just diverse attack content) can close the remaining word-substitution attack gap found in the ablation
Address the homoglyph confusables limitation properly using a dedicated mapping table rather than the partial hand-built map

### Deployment/production-facing:

Package the DistilBERT + preprocessing pipeline as a lightweight API endpoint (FastAPI) that other applications could call before passing user input to an LLM
Add usage monitoring/logging to the deployed app to track false positive/negative patterns on real traffic over time

### Research-facing:

Investigate whether the non-deterministic training variance observed (from Python's set() hash randomization) is worth controlling for via a fixed PYTHONHASHSEED, for more reproducible benchmarking
Extend the "does training data diversity fix vulnerabilities better than preprocessing" finding into a more systematic study by testing on additional held-out datasets beyond HackAPrompt

### Individual:
DistilBERT: 
Close the remaining word-substitution attack gap (BAE, PWWS, TextFooler, Bert-Attack, etc. — still evaded at low but nonzero rates) by fine-tuning directly on adversarially-perturbed training examples, rather than relying on training-data diversity alone.
- Replace the hand-built homoglyph map with a complete confusables mapping table (e.g. via the `confusable_homoglyphs` package), since the current map only covers common Cyrillic/Greek lookalikes and misses less frequent ones.
- Re-run training with a fixed `PYTHONHASHSEED` to eliminate the run-to-run variance observed from Python's non-deterministic `set()` ordering, for more reproducible benchmarking.
- Move from a single train/test split to k-fold cross-validation for a more robust estimate of model performance, rather than relying on one fixed split.
- Apply dynamic quantization to the deployed model to reduce inference latency for users interacting with the live app, without materially affecting accuracy.
- Extend generalization testing beyond HackAPrompt to at least one additional independently sourced attack dataset, to build more confidence that the Tensor Trust retraining genuinely improved real-world robustness rather than overfitting to that specific dataset's style.
