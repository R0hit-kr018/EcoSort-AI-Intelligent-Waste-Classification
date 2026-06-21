# ♻️ EcoSort AI — Intelligent Waste Classification

> **AI-powered waste classification system** built with YOLOv8 and IBM watsonx, featuring a rich Streamlit dashboard.

[![Python](https://img.shields.io/badge/Python-3.14-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.58-red?logo=streamlit)](https://streamlit.io)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Nano-green?logo=ultralytics)](https://ultralytics.com)
[![IBM watsonx](https://img.shields.io/badge/IBM-watsonx%20AI-blue?logo=ibm)](https://ibm.com/watsonx)

---

## 🌍 Overview

**EcoSort AI** uses computer vision to instantly classify waste into 6 categories and provides:
- ♻️ Proper disposal methods
- 🌿 Environmental impact data
- 💡 Sustainability recommendations
- 🎯 UN SDG alignment (SDG 12 & 13)
- 🤖 IBM watsonx AI-generated sustainability advice

---

## 🖼️ Screenshots

| Home Page | Dashboard |
|-----------|-----------|
| Dark green theme with stats & Plotly charts | Model performance bar + dataset donut chart |

---

## 🗂️ Project Structure

```
EcoSort_AI/
├── app/
│   ├── app.py                  # Main Streamlit application
│   ├── granite_helper.py       # IBM watsonx Granite integration
│   ├── waste_knowledge.py      # Waste info database
│   └── .streamlit/
│       └── config.toml         # Dark theme configuration
├── models/
│   └── best.pt                 # YOLOv8 trained model (not tracked in git)
├── train.py                    # Model training script
├── predict.py                  # Standalone prediction script
├── requirements.txt            # Python dependencies
└── .env.example                # Environment variables template
```

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/R0hit-kr018/EcoSort-AI-Intelligent-Waste-Classification.git
cd EcoSort-AI-Intelligent-Waste-Classification
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up environment variables
```bash
cp .env.example .env
# Fill in your IBM watsonx credentials in .env
```

### 4. Add your trained model
Place your trained `best.pt` file in the `models/` directory.

### 5. Run the app
```bash
streamlit run app/app.py
```

Open `http://localhost:8501` in your browser.

---

## 🔑 Environment Variables

Create a `.env` file in the project root:

```env
IBM_API_KEY=your_ibm_api_key_here
IBM_PROJECT_ID=your_project_id_here
IBM_URL=https://us-south.ml.cloud.ibm.com
```

---

## 🤖 Model Details

| Property | Value |
|----------|-------|
| Architecture | YOLOv8 Nano Classification |
| Parameters | 1.45M (lightweight) |
| Accuracy | 88.4% |
| Precision | 87% |
| Recall | 86% |
| F1-Score | 86.5% |
| Training Images | 2,019 |
| Epochs | 20 |

---

## 🗃️ Waste Categories

| Category | Emoji | Sustainability Score | Decomposition |
|----------|-------|---------------------|---------------|
| Cardboard | 📦 | 88/100 | ~60 days |
| Glass | 🍾 | 95/100 | 1,000,000+ years |
| Metal | 🥫 | 92/100 | 200–500 years |
| Paper | 📄 | 90/100 | ~5–6 weeks |
| Plastic | 🧴 | 75/100 | 400–1,000 years |
| Trash | 🗑️ | 40/100 | 20–30 years avg |

---

## 🏗️ Tech Stack

- **Computer Vision:** YOLOv8 (Ultralytics)
- **AI Advisor:** IBM watsonx — Llama 3.3 70B Instruct
- **Frontend:** Streamlit 1.58 with custom CSS
- **Charts:** Plotly
- **Image Processing:** PIL / OpenCV
- **Data:** Pandas, NumPy

---

## 🎯 SDG Alignment

- **SDG 12** — Responsible Consumption and Production
- **SDG 13** — Climate Action

---

## 📄 License

MIT License — feel free to use, modify, and distribute.

---

<div align="center">
  <b>♻️ EcoSort AI · Empowering Sustainable Waste Management Through AI</b>
</div>
