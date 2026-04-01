# 🩺 AB-GEN Medical: Trusted XAI for Hematological Diagnosis
> **Technical Prospectus & Clinical Demonstration**

---

## 📝 Abstract
The adoption of Artificial Intelligence in clinical environments is often hindered by the "Black Box" nature of modern Deep Learning models. **AB-GEN Medical** introduces a paradigm shift in hematological diagnosis by providing a **traceable, interpretable, and high-performance** system for blood cell classification. Using a multi-stage geometric and spectral inference engine, AB-GEN identifies 8 types of blood cells while generating a comprehensive 7-step clinical report and real-time activation heatmaps for every single diagnosis.

---

## 📺 Video Demonstration
> [!IMPORTANT]
> **[Watch the AB-GEN Medical Demo on YouTube](https://youtu.be/gMhgHYC_c1Q?is=NOdAioIA9DDgscwq)**

---

## 💡 The Problem: Medical "Black Boxes"
Current AI models in medicine often provide a verdict (e.g., "98% Lymphocyte") without explaining the underlying morphological rationale. In high-stakes clinical decisions, this lack of transparency is unacceptable.

### Our Solution: Inherent Traceability
AB-GEN Medical was built from the ground up to be **Explainable by Design**. Instead of end-to-end black box training, our **Quimera V24** architecture follows a rigorous clinical reasoning process:
1. **Morphological Extraction**: Captures 99.2% of visual information.
2. **Fisher Discriminant Analysis**: Amplifies the most relevant diagnostic features.
3. **Topological Frequency Analysis**: Studies the "spectral signature" of the cell.
4. **Geometric Prototyping**: Measures affinity against established class centroids.
5. **Class Consensus**: A swarm of independent specialists (MLPs and LogReg) analyzes the sample.
6. **Second-Level Verification**: A meta-learner validates the consensus and quantifies certainty.

---

## 🔬 XAI Traceability Report
Every diagnosis in this demo includes:
- **Clinical Trace Visibility**: A 7-step breakdown of the AI's reasoning.
- **Topological Heatmaps**: Direct visualization of the cellular regions that triggered the diagnosis.
- **Calibrated Certainty**: Full probability distribution over 8 cell classes.

---

## 🌿 Green AI: Performance & Sustainability (M5)
AB-GEN Medical is part of the **M5 initiative**, prioritizing efficient computing without sacrificing accuracy.
- **Latency**: < 100ms per sample.
- **Precision**: 87.14% (Tested on BloodMNIST).
- **Eco-Impact**: Ultra-low energy consumption and CO₂ footprint per diagnosis.

---

## 🚀 Getting Started (Testing the Demo)

### 1. Prerequisites
- Python 3.10+
- Models: `med_pca_cache.npz`, `med_quimera_n1_elite.pkl`, `med_quimera_n2_poly_elite.pkl` (Must be placed in the parent directory of this demo).

### 2. Quick Launch
```bash
git clone https://github.com/your-username/AB-GEN-MEDICAL-DEMO.git
cd AB-GEN-MEDICAL-DEMO
pip install -r requirements_demo.txt
python server.py
```
Or simply double-click **`run_demo.bat`** on Windows.

### 3. Access the Dashboard
Open your browser at **http://localhost:5000** to explore the interactive clinical dashboard.

---

## 📁 Technical Architecture
The engine consists of a "Blind Inference" pipeline separated from the training core:
- **`engine.py`**: Handles Fourier transforms, PCA projections, and XAI heatmap generation.
- **`server.py`**: Flask-based API for seamless dashboard integration.
- **`app.js`**: Managed visualization of probability tensors and clinical reports.

---

## 📄 License & Disclaimer
- **Code**: MIT License.
- **Models & Architecture**: Proprietary and Confidential (AB-GEN).
- **Disclaimer**: This is a research demonstration tool. Not for real-time medical diagnosis in professional clinical settings.

---

<p align="center">
  <strong>AB-GEN Medical © 2026</strong><br>
  Leading the way in Traceable Clinical Intelligence.
</p>
