<div align="center">

<img src="https://img.shields.io/badge/NanoGNN--X-Atomistic%20AI-FF6B6B?style=for-the-badge&logo=atom&logoColor=white" alt="NanoGNN-X"/>

# NanoGNN-X
### *Explainable AI for Nanomaterial Discovery*

**Physics-aware Graph Neural Network that predicts electronic band gaps & formation energies of crystalline materials — in under 100ms.**

[![Live Demo](https://img.shields.io/badge/🌐%20Live%20Demo-nanognn--x.web.app-FF6B6B?style=for-the-badge)](https://nanognn-x.web.app)
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-GNN%20Engine-EE4C2C?style=for-the-badge&logo=pytorch)](https://pytorch.org)
[![ONNX](https://img.shields.io/badge/ONNX-Runtime-005CED?style=for-the-badge&logo=onnx)](https://onnxruntime.ai)
[![CI](https://img.shields.io/github/actions/workflow/status/SAMAR106/NanoGnn-x/ci.yml?branch=main&style=for-the-badge&label=CI)](https://github.com/SAMAR106/NanoGnn-x/actions)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

---

<img width="900" alt="NanoGNN-X Dashboard" src="https://nanognn-x.web.app/og.png" onerror="this.style.display='none'"/>

> **"Discovering new materials with quantum-level accuracy at silicon speed."**

</div>

---

## 🧬 What is NanoGNN-X?

Traditional material discovery takes **years** in a laboratory. NanoGNN-X accelerates this process by orders of magnitude using a **Three-Body Atomistic Line Graph Neural Network (ALIGNN)** that understands the true geometry of crystal structures — not just atom types, but the precise distances and **angles** between bonds.

Upload any crystal structure file (`.cif`) and within **milliseconds** receive:
- ⚡ **Electronic Band Gap** prediction (eV)
- 🔋 **Formation Energy** prediction (eV/atom)
- 🧠 **Explainability (XAI) Heatmap** — showing *which atoms* drove the prediction

---

## ✨ Key Features

| Feature | Description |
|---|---|
| ⚛️ **Physics-Aware GNN** | ALIGNN architecture with Bessel RBF & Chebyshev angle basis |
| 🚀 **~100ms Inference** | ONNX Runtime / TensorRT compiled model |
| 🔥 **XAI Heatmap** | Atom-level ablation-based saliency — red = important, blue = ignored |
| 📊 **Interactive Dashboard** | Live Parity Plot, Loss Curves, Feature Importance charts |
| 🔬 **Material Explorer** | Search & browse the full JARVIS-DFT training dataset |
| 📦 **Batch Processing** | Drop multiple `.cif` files — get a full CSV report instantly |
| 🏭 **MLOps Loop** | Flag uncertain predictions for DFT re-verification & retraining |
| 🌐 **Production REST API** | FastAPI with auth, rate-limiting, CORS, typed responses |
| 🐳 **Docker Ready** | Single `docker-compose up` spins up the full stack |

---

## 🏗️ Architecture

```
┌────────────────────────────────────────────────────────┐
│                   NanoGNN-X Platform                    │
├──────────────┬─────────────────────┬───────────────────┤
│   Frontend   │     API Layer       │   ML Engine       │
│  React/Vite  │   FastAPI + CORS    │  ALIGNN PyTorch   │
│  Recharts    │   /predict          │  → ONNX Export    │
│  3Dmol.js    │   /predict/batch    │  → ONNXRuntime    │
│  Firebase    │   /explain (XAI)    │  → TensorRT (GPU) │
│  Hosting     │   /dataset          │                   │
└──────────────┴─────────────────────┴───────────────────┘
```

### The Three-Body GNN Physics Engine

Unlike naive neural networks that treat atoms as isolated points, NanoGNN-X models **true crystal geometry**:

```
Atom Features (Z, χ, r_cov, IE ...)
        ↓
   Node Embeddings
        ↓
Bond Distance → BesselRBF(d_ij)  ──┐
Bond Angle  → ChebyshevBasis(θ)  ──┤→  ALIGNN Message Passing
                                    │    (captures 3-body forces)
                                    ↓
                              Graph Pooling
                                    ↓
                        [Band Gap, Formation Energy]
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Node.js 18+
- CUDA GPU (optional, for TensorRT acceleration)

### 1. Clone & Install

```bash
git clone https://github.com/SAMAR106/NanoGnn-x.git
cd NanoGnn-x
pip install -r requirements.txt
```

### 2. Start the Backend API

```bash
uvicorn api.server:app --host 0.0.0.0 --port 8000
```

The API is now live at `http://localhost:8000`. Test it:

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@data/raw_cifs/NaCl.cif"
```

**Response:**
```json
{
  "bandgap_eV": 4.336,
  "formation_energy_eV_per_atom": -2.149,
  "inference_time_ms": 102.3,
  "backend": "onnxruntime"
}
```

### 3. Start the Frontend

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173` — the full dashboard is live locally.

### 4. Docker (Full Stack)

```bash
docker-compose up --build
```

---

## 🧪 Running Tests

```bash
# Full test suite
pytest tests/ -v

# Type checking
mypy api/ src/ train.py

# Lint
ruff check .
```

---

## 📊 Model Performance

| Metric | Value |
|---|---|
| **MAE (Band Gap)** | 0.184 eV |
| **RMSE (Band Gap)** | 0.241 eV |
| **R² Score** | 0.912 |
| **Inference Speed** | ~100ms (ONNX) / ~15ms (TensorRT) |
| **Training Dataset** | 12,458 JARVIS-DFT structures |
| **Train / Val / Test Split** | 70% / 15% / 15% |

> These results are on par with state-of-the-art ALIGNN models reported in the literature (Choudhary et al., 2021).

---

## 🧠 Explainability (XAI) Pipeline

NanoGNN-X is **not a black box**. Our XAI pipeline uses a rapid ablation method:

1. Run baseline prediction on the full crystal graph.
2. Mask each atom's embedding vector to zero, one at a time.
3. Measure how much the Band Gap prediction shifts.
4. Normalize the shift values → **saliency score per atom**.
5. Map scores to a **Red (hot) → Blue (cold)** color gradient on the 3D viewer.

This tells scientists exactly **which atomic sites** are driving the electronic properties — critical for designing better materials.

---

## 🗂️ Project Structure

```
nanognn-x/
│
├── api/                    # FastAPI REST API
│   ├── server.py           # Main app: /predict, /explain, /batch, /dataset
│   └── engine.py           # ONNX / TensorRT inference engine
│
├── src/                    # Core ML library
│   ├── model.py            # ALIGNN GNN architecture
│   ├── dataset.py          # CIF → PyG graph conversion
│   ├── config.py           # Centralized config with pathlib paths
│   └── layers.py           # BesselRBF, ChebyshevBasis, message passing
│
├── frontend/               # React + Vite dashboard
│   ├── src/
│   │   ├── App.jsx         # Main dashboard (Dashboard, Predict, Explorer tabs)
│   │   └── index.css       # Design system
│   └── firebase.json       # Firebase Hosting config
│
├── scripts/                # Utility scripts
│   ├── export_onnx.py      # PyTorch → ONNX export
│   └── validate_tensorrt.py # TensorRT benchmark (requires GPU)
│
├── tests/                  # Pytest test suite
├── notebooks/              # Hackathon Presentation notebook
├── checkpoints/            # Trained model weights (ONNX)
├── train.py                # Training entry point (Lightning)
├── Dockerfile              # Production container
├── pyproject.toml          # Project metadata & tool config
└── requirements.txt        # Python dependencies
```

---

## 🔌 API Reference

### `POST /predict`
Predict band gap and formation energy for a single CIF file.

| Parameter | Type | Description |
|---|---|---|
| `file` | `multipart/form-data` | Crystal structure `.cif` file |

### `POST /predict/batch`
Process multiple structures simultaneously.

| Parameter | Type | Description |
|---|---|---|
| `files` | `multipart/form-data` | Multiple `.cif` files |

### `POST /explain`
Generate atom-level XAI saliency heatmap.

| Parameter | Type | Description |
|---|---|---|
| `file` | `multipart/form-data` | Single `.cif` file |

**Response:**
```json
{
  "heatmap": [0.87, 0.12, 0.95, 0.43, ...],
  "justification": "The model's prediction is strongly driven by the Na atoms..."
}
```

### `GET /health`
Service health check. Returns backend type and normalizer status.

---

## 🌐 Live Demo

The frontend is deployed on **Firebase Hosting**:

> **🔗 [https://nanognn-x.web.app](https://nanognn-x.web.app)**

The Dashboard tab (metrics, plots) works fully on the live link. The Predict & XAI features require the FastAPI backend to be running (see Quick Start above).

---

## 🛣️ Roadmap

- [ ] **Cloud Backend Deployment** — Deploy FastAPI to Google Cloud Run for a fully serverless, public prediction API
- [ ] **Active Learning Loop** — Automatically flag high-uncertainty predictions for DFT re-simulation
- [ ] **Edge AI (Browser Inference)** — Load the ONNX model directly in the browser via `onnxruntime-web` for zero-latency, offline predictions
- [ ] **Multi-Property Prediction** — Extend targets to include thermal conductivity, elastic moduli, and optical properties
- [ ] **Generative Materials Discovery** — Use a GNN-based generative model to propose novel crystal structures with target band gaps

---

## 📚 References

- Choudhary, K., & DeCost, B. (2021). **Atomistic Line Graph Neural Network for improved materials property predictions.** *npj Computational Materials*, 7(1), 185. [DOI](https://doi.org/10.1038/s41524-021-00650-1)
- Gasteiger, J., et al. (2020). **Directional Message Passing for Molecular Graphs.** *ICLR 2020.*
- JARVIS-DFT Database: [https://jarvis.nist.gov](https://jarvis.nist.gov)
- Materials Project: [https://materialsproject.org](https://materialsproject.org)

---

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

<div align="center">

**Built with ❤️ for the hackathon — accelerating the future of nanomaterial discovery.**

[![GitHub stars](https://img.shields.io/github/stars/SAMAR106/NanoGnn-x?style=social)](https://github.com/SAMAR106/NanoGnn-x/stargazers)

</div>
