<div align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&ColorList=896A15,6E8915,98351F&height=200&section=header&text=BBBP%20%7C%20Molecular%20Reasoning&fontSize=50&fontColor=DEE2D9&animation=fadeIn&fontAlignY=35&desc=Mistral-7B%20%7C%20Property-Aware%20Fine-tuning%20%7C%20Best%20ROC-AUC:%200.7141&descAlignY=60&descAlign=50" width="100%"/>
</div>

<div align="center">
  <img 
    src="https://readme-typing-svg.herokuapp.com?font=Fira+Code&weight=600&size=22&pause=1000&color=F0C419&center=true&vCenter=true&width=1000&lines=Fine-tuning+Mistral-7B+on+BBBP+Benchmark;Best+ROC-AUC:+0.7141+on+Scaffold+Split;RDKit+Nitrogen+Valence+Sanitization;Property-Aware+Prompting+(MW+%26+TPSA)."
    alt="Typing SVG"
  />
</div>

<br/>

## 🧬 Project Mission

This repository contains the scientific implementation of a **Mistral-7B** based Blood-Brain Barrier Penetration (BBBP) predictor. Developed for **DeepChem GSoC 2026**, this project moves beyond standard SMILES-only fine-tuning by injecting physical chemical descriptors directly into the prompt — giving the LLM topological context it otherwise lacks.

---

## 📊 Results — All Experiments

| Experiment | Strategy | ROC-AUC | Split |
|------------|----------|---------|-------|
| `bbbp-benchmark-code.ipynb` | Step-based, peak at Step 1000 | 0.7556 | Scaffold |
| `mistral-bbbp.ipynb` Exp 1 | Epoch-based, 5 epochs | 0.6904 | Scaffold |
| `mistral-bbbp.ipynb` Exp 2 | Augmented prompts + 3 epochs | **0.7141** ✅ | Scaffold |

**Final Best: 0.7141 ROC-AUC — Scaffold Split, Kaggle T4**

Training curve (Experiment 2 — Final):

| Epoch | ROC-AUC | Status |
|-------|---------|--------|
| 1 | 0.7018 | New record ↑ |
| 2 | 0.6687 | Overfitting ↓ |
| 3 | **0.7141** | **BEST** ✅ |

---

## 🔬 Key Scientific Contributions

| Feature | Implementation | Impact |
|---------|---------------|--------|
| **Data Sanitization** | RDKit valence filtering | Removed invalid Nitrogen valence molecules before tokenization |
| **Property-Aware Prompting** | MW + TPSA injected into prompt | LLM receives 2D topology context alongside 1D SMILES |
| **Architecture** | LoRA r=32, 4-bit NF4 | Fits Mistral-7B in 16GB VRAM |
| **Evaluation** | Scaffold Split | True out-of-distribution generalization |

**Prompt format used:**
```
Analysis: Brain Barrier Penetration
Descriptors: MW=310.4, TPSA=45.2
SMILES: Cc1ccc(cc1)...
Permeable: Yes
```

This is a lightweight preview of the **GIMF architecture** proposed for GSoC — injecting molecular descriptors into LLM context to bridge 1D text and 2D topology.

---

## 🛠️ Tech Stack

<div align="center">
  <img src="https://img.shields.io/badge/DeepChem-Scientific_AI-blue?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black"/>
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/LoRA-PEFT-green?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/RDKit-Chemistry-red?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Mistral--7B-4bit_NF4-purple?style=for-the-badge"/>
</div>

---

## 📂 File Index

| File | Description |
|------|-------------|
| `bbbp-benchmark-code.ipynb` | Step-based training, peak ROC-AUC 0.7556 |
| `mistral-bbbp.ipynb` | Epoch-based with augmented prompts, best **0.7141** |
| `requirements.txt` | Environment dependencies |

---

## 💻 How to Run
```bash
pip install -r requirements.txt

# Step-based experiment
# Open bbbp-benchmark-code.ipynb in Kaggle

# Epoch-based best result — Final
# Open mistral-bbbp.ipynb in Kaggle
# kaggle.com/sameernadeem66/mistral-bbbp
```

---

## 🔗 Part of DeepChem GSoC 2026 Research

| Task | Model | Result | Repo |
|------|-------|--------|------|
| BACE Classification | Mistral-7B QLoRA | 0.8371 ROC-AUC | [BACE Repo](https://github.com/Abu-Sameer-66/Mistral7B-BACE-Generalization-Study) |
| **BBBP Classification** | **Mistral-7B QLoRA** | **0.7141 ROC-AUC** | This Repo |
| ClinTox Classification | Mistral-7B QLoRA | 0.9913 ROC-AUC | [ClinTox Repo](https://github.com/Abu-Sameer-66/Mistral7B-ClinTox-Study) |
| Tox21 Analysis | RF + ECFP + OLMo | Scaffold gap proof | [Tox21 Repo](https://github.com/Abu-Sameer-66/Mistral7B-Tox21-Molecular-Optimization) |
| ESOL Regression | OLMo-7B + Reg Head | 0.8582 RMSE | [ESOL Repo] |
| SMILES Generation | OLMo-7B + RDKit TSM | 20/20 = 100% valid | [Generation Repo](https://github.com/Abu-Sameer-66/olmo-smiles-generation-tsm) |
