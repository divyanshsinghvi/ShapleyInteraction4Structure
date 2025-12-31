# 📚 Using Shapley Interactions to Understand How Models Use Structure

[![arXiv](https://img.shields.io/badge/arXiv-2403.13106-b31b1b.svg)](https://arxiv.org/abs/2403.13106)
[![ACL 2025](https://img.shields.io/badge/ACL--2025-Paper-blue)](https://aclanthology.org/2025.acl-long.1011)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

This repository provides code and scripts to reproduce experiments from the paper:

> **“Using Shapley Interactions to Understand How Models Use Structure”**  
> _Divyansh Singhvi, Diganta Misra, Andrej Erkelens, Raghav Jain, Isabel Papadimitriou, Naomi Saphra_  
> [arXiv:2403.13106v2](https://arxiv.org/abs/2403.13106v2) · [ACL 2025 Paper](https://aclanthology.org/2025.acl-long.1011)

---

## 📝 Overview

Modern language and speech models learn rich hidden structures about **syntax**, **semantics**, and **phonetics**.

**This repository shows how to use the Shapley-Taylor Interaction Index (STII)** to quantify **pairwise interactions**:
- **Text models:** How do pairs of tokens interact beyond their individual effects?
- **Speech models:** How do acoustic frames interact near phoneme boundaries?

By doing so, you can test:
- How well models encode **syntactic tree structures**
- How they handle **multiword expressions**
- How speech models reflect **phonetic coarticulation**

---

## 📂 Repository Structure


---

## 🧮 How It Works

✅ **STII for Text (`ExperimentRunner`):**  
- Load tagged sentences with **multiword expressions (MWEs)** and **syntactic trees**  
- For token pairs:
  - Compute logits for 4 contexts: `AB`, `A`, `B`, `φ` (none)
  - Interaction = `(AB - A - B + φ)` and normalize by `(φ)` norms
- Analyze how interaction varies with:
  - **Linear distance**
  - **Syntactic distance**
  - Whether tokens belong to a **strong or weak MWE**

✅ **STII for Speech (`SpeechSTIIExperimentRunner`):**  
- Load audio and phoneme time alignments
- Mask 20ms waveform slices to simulate ablations
- Compare interaction:
  - **Consonant-vowel** vs **consonant-consonant**
  - By manner of articulation (how vowel-like a consonant is)
  - The methodology is same for both Speech and Text

---

## 🚀 How to Run

### 1️⃣ Install Environment

Create the environment using `conda.yaml`:

```bash
conda env create -f conda.yaml -n shapley_llm
conda activate shapley_llm
```

### 2️⃣ Run Text Experiments

1. Generate Language Data
```bash
cd language/mwe_tagger
./run_sr_pipeline.sh <model_name> <model_name>
```

model_names : ['gpt2', 'bert']

Places the MWE tagger outputs at (`bert_bert.pkl_*` or `gpt_gpt.pkl_*`) in `language/mwe_tagger/`
Then run:

```bash
python language/language_runner.py
```

### 3️⃣ Run Speech Experiments

Add audio files to `speech_data/mfa_inp_new/` and phoneme CSVs to `speech_data/extracted_phonemes/`. Results are written to `speech_data/stii_outputs_fix/`.
To generate the files and csv use files at speech/data_utils/ 


```bash
python speech/speech_runner.py
```
