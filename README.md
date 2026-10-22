# AB-GEN Medical — Experimental Hematology Image Demo

**[Ficha verificable del proyecto](docs/PROJECT_STATUS.md)** · Esta demostración no dispone de validación clínica para decisiones sobre pacientes.
> **RESEARCH PROTOTYPE — NOT A MEDICAL DEVICE. NOT FOR DIAGNOSIS OR TREATMENT.**


See [Evidence and limitations](EVIDENCE_STATUS.md), [Security](SECURITY.md) and [Contributing](CONTRIBUTING.md) for reproduction requirements.

---

## 📝 Abstract
The adoption of Artificial Intelligence in clinical environments is often hindered by the "Black Box" nature of modern Deep Learning models. **AB-GEN Medical** explores a traceable multi-stage image-classification interface. The diagnostic accuracy and calibration of this demonstration have not been independently established. Using a multi-stage geometric and spectral inference engine, The UI presents eight example cell categories with an exploratory seven-stage technical trace and illustrative activation visualizations; these are not clinical reports or validated diagnoses.

---

## 📺 Video Demonstration
> [!IMPORTANT]
> **[Watch the AB-GEN Medical Demo on YouTube](https://youtu.be/gMhgHYC_c1Q?is=NOdAioIA9DDgscwq)**

---

## 💡 The Problem: Medical "Black Boxes"
Current AI models in medicine often provide a verdict (e.g., "98% Lymphocyte") without explaining the underlying morphological rationale. In high-stakes clinical decisions, this lack of transparency is unacceptable.

### Experimental Technical Pipeline
AB-GEN Medical was built from the ground up to be **Explainable by Design**. Instead of end-to-end black box training, our **Quimera V24** architecture presents a sequence of engineering transformations:
1. **Morphological Extraction**: Performs dimensionality reduction; the historical 99.2% figure is not independently validated here.
2. **Fisher Discriminant Analysis**: Amplifies the most relevant diagnostic features.
3. **Topological Frequency Analysis**: Studies the "spectral signature" of the cell.
4. **Geometric Prototyping**: Measures affinity against established class centroids.
5. **Class Consensus**: A swarm of independent specialists (MLPs and LogReg) analyzes the sample.
6. **Second-Level Verification**: A meta-learner combines component outputs; confidence calibration has not been clinically validated.

---

## 🔬 XAI Traceability Report
Each example classification in this research interface can include:
- **Engineering trace visibility**: a seven-stage display of calculated model features.
- **Exploratory heatmaps**: visualizations whose clinical faithfulness has not been established.
- **Unvalidated scores**: eight output scores; do not interpret them as clinically calibrated probabilities.

---

## Green AI — Historical Figures Awaiting Reproduction
AB-GEN Medical is part of the **M5 initiative**, prioritizing efficient computing without sacrificing accuracy.
- **Latency**: historical claim of below 100 ms, not reproduced here with hardware/measurement protocol.
- **Accuracy**: 87.14% historically reported for BloodMNIST; test split, preprocessing and source artifact reproducibility need independent audit.
- **Energy/CO₂**: no independently verified comparative measurements are provided; no quantitative advantage is claimed.

---

## 🚀 Getting Started (Testing the Demo)

### 1. Prerequisites
- Python 3.10+
- Models: `med_pca_cache.npz`, `med_quimera_n1_elite.pkl`, `med_quimera_n2_poly_elite.pkl` (Must be placed in the parent directory of this demo).

### 2. Quick Launch
```bash
git clone https://github.com/msrobotika/AB-GEN-MEDICAL-DEMO..git
cd AB-GEN-MEDICAL-DEMO.
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
- **Code**: MIT License, see [LICENSE](LICENSE); model files and any data retain their separate rights.
- **Models & Architecture**: Proprietary and Confidential (AB-GEN).
- **Disclaimer**: Research interface only, not for real-world medical diagnosis, triage, treatment, or clinical risk stratification. No identifiable patient material should be uploaded.

---

<p align="center">
  <strong>AB-GEN Medical © 2026</strong><br>
  Experimental interpretable image-classification research.
</p>
