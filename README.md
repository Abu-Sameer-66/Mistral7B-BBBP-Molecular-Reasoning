<div align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&ColorList=896A15,6E8915,98351F&height=200&section=header&text=BBBP%20%7C%20Molecular%20Reasoning&fontSize=50&fontColor=DEE2D9&animation=fadeIn&fontAlignY=35&desc=Mistral-7B%20%7C%20Property-Aware%20Fine-tuning%20%7C%20GSoC%20'26&descAlignY=60&descAlign=50" width="100%"/>
</div>

<div align="center">
  <img 
    src="https://readme-typing-svg.herokuapp.com?font=Fira+Code&weight=600&size=22&pause=1000&color=F0C419&center=true&vCenter=true&width=1000&lines=Fine-tuning+Mistral-7B+on+BBBP+Benchmark;Achieving+0.7556+Peak+ROC-AUC;Nitrogen+Valence+Sanitization+Implemented;Property-Aware+Prompting+(MW+%26+TPSA)."
    alt="Typing SVG"
  />
</div>

<br/>

---















### 🧬 Project Mission: Beyond Simple Classification
This repository contains the scientific implementation of a **Mistral-7B** based Blood-Brain Barrier Penetration (BBBP) predictor. Developed for **DeepChem GSoC 2026**, this project moves beyond standard "SMILES-only" fine-tuning by inducing physical chemical reasoning through property-aware descriptors.

---

### 🚀 Technical Excellence & Innovation

| Feature | Implementation | Scientific Impact |
|:---|:---|:---|
| **Data Sanitization** | **Valence Filtering**. | Identified and removed 11 molecules with invalid Nitrogen valences (> 4) to prevent numerical instability. |
| **Reasoning Engine** | **Property-Aware Prompting**. | Integrated MW and TPSA descriptors into prompts to guide the model's latent reasoning on passive diffusion. |
| **Architecture** | **LoRA Rank $r=32$**. | Higher rank adapter to capture the complex structural features required for barrier permeability. |
| **Efficiency** | **4-bit NF4 Quantization**. | Memory-optimized for 16GB VRAM GPUs (T4 Colab) with a VRAM footprint of under 6GB. |

---

### 📊 Performance Benchmarks (BBBP)

- **Metric:** Peak Mean ROC-AUC of **0.7556** achieved at Step 1000.
- **Validation Strategy:** Rigorous **Scaffold Splitting** methodology to ensure out-of-distribution structural generalization.

![BBBP Optimization Curve](bbbp_optimization_curve.png)
*Figure: The optimization arc demonstrates stable learning via gradient norm clipping and property-aware context.*

---

### 🛠️ Production Tech Stack
<div align="center">
  <img src="https://img.shields.io/badge/DeepChem-Scientific_AI-blue?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black"/>
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/LoRA-PEFT-green?style=for-the-badge"/>
</div>

---

### 📂 Repository Structure
- `bbbp_benchmark.py`: Core fine-tuning script with property-aware logic.
- `requirements.txt`: Environment dependencies.
- `bbbp_optimization_curve.png`: Scientific visualization of the learning arc.

### 💻 How to Run
```bash
# Install dependencies
pip install -r requirements.txt

# Execute the property-aware benchmark
python bbbp_benchmark.py
