# ConU: Conformal Uncertainty in Large Language Models with Correctness Coverage Guarantees

**Authors:** Zhiyuan Wang, Jinhao Duan, Lu Cheng, Yue Zhang, Qingni Wang, Xiaoshuang Shi, Kaidi Xu, Hengtao Shen, Xiaofeng Zhu

**Venue:** EMNLP 2024 Findings. arXiv:2407.00499, June 2024 (revised November 2024)

**Category:** Uncertainty Quantification / Conformal Prediction

## Abstract

UQ in natural language generation tasks remains an open challenge, exacerbated by the closed-source nature of the latest LLMs. This study applies conformal prediction to black-box LLMs in open-ended NLG tasks. The authors introduce a self-consistency-based uncertainty metric and develop a conformal uncertainty method that integrates correctness alignment into the prediction framework. Testing across seven LLMs and four free-form datasets (general and medical) demonstrates strict control over the correctness coverage rate while producing calibrated prediction sets with small size.

## Key Contribution

First systematic application of conformal prediction to open-ended NLG with black-box LLMs, providing finite-sample statistical coverage guarantees on answer correctness.

## Relevance to Gas Town / MAUQ

ConU's conformal prediction framework could provide MAUQ with formal statistical guarantees (coverage guarantees) that the current entropy/PMI/DAG approaches lack. Wrapping MAUQ's uncertainty scores in a conformal prediction framework would give users provable coverage bounds -- a strong scientific contribution direction.
