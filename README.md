# DeepCare: Patient Dropout Prediction System

## 📋 Overview

**DeepCare** is an end-to-end machine learning pipeline that predicts patient dropout risk in healthcare settings using advanced data mining, feature engineering, and Bayesian optimization. The system processes patient demographics, hospital visit logs, and medical records to identify at-risk patients and enable proactive interventions.

### Key Features
- 🔍 **Multi-source Data Integration**: Combines patient demographics, visit history, and hospital logs
- ⚙️ **Automated Feature Engineering**: Intelligent log-derived feature extraction
- 🎯 **Patient Segmentation**: K-means clustering for cohort-based analysis
- ⚖️ **Class Balancing**: Handles imbalanced datasets with strategic oversampling
- 🔧 **Bayesian Optimization**: Automated hyperparameter tuning via Optuna
- 🚀 **Production-Ready APIs**: FastAPI & Streamlit interfaces for predictions
- 🐳 **Containerized Deployment**: Docker & CI/CD with GitHub Actions

---

## 🏗️ Architecture

### ML Pipeline Flow
```
Raw Data (CSV/XML) 
    ↓
Load & Merge (data_loader.py)
    ↓
Feature Engineering (preprocess.py)
    ↓
Encode & Scale (preprocess.py)
    ↓
Patient Segmentation (mining.py)
    ↓
Class Balancing (balancing.py)
    ↓
Model Training with HPO (train.py)
    ↓
Evaluation & Metrics (evaluate.py)
    ↓
Batch Predictions (predict.py)
```

### Project Structure
```
DeepCare/
├── src/                          # Core ML pipeline
│   ├── main.py                   # 🎯 Driver script - orchestrates entire pipeline
│   ├── data_loader.py            # Load & merge patient demographics, visits, hospital logs
│   ├── preprocess.py             # Feature engineering, encoding, scaling
│   ├── mining.py                 # K-means clustering for patient segmentation
│   ├── balancing.py              # Handle class imbalance via oversampling
│   ├── train.py                  # Model training with Optuna-based hyperparameter tuning
│   ├── evaluate.py               # Model evaluation & metrics computation
│   ├── predict.py                # Batch & real-time prediction generation
│   ├── tune.py                   # Advanced tuning utilities
│   ├── config.py                 # Configuration parser (YAML)
│   └── logger.py                 # Centralized logging utility
├── app/                          # Deployment interfaces
│   ├── app.py                    # FastAPI backend for REST predictions
│   └── serve.py                  # Streamlit frontend for interactive UI
├── configs/
│   └── config.yaml               # Pipeline configuration (data paths, model params)
├── notebooks/
│   └── deepcare.ipynb            # Full exploratory workflow & analysis
├── docker-compose.yaml           # Multi-service containerization (FastAPI + Streamlit)
├── Dockerfile                    # Container image definition
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

## 📊 Data Requirements

The dataset files are **not included** in this repository due to size and privacy constraints (HIPAA compliance).

### Download Dataset
📥 **[Download from Google Drive](https://drive.google.com/drive/folders/1HT7y8O2PJTfMkYw8u0K_1e4_KcYyvaN5?usp=sharing)**

### Expected Data Structure
Place downloaded files in the `data/` folder:
```
data/
├── patient_demographics.csv      # Patient ID, age, gender, medical history
├── patient_visits.csv            # Patient ID, visit dates, visit outcomes
└── hospital_logs.xml             # Patient ID, admission logs, discharge details
```

---

## 🚀 Quick Start

### 1️⃣ Prerequisites
- Python 3.10+
- Docker & Docker Compose (optional, for containerized deployment)
- Git

### 2️⃣ Local Setup

```bash
# Clone repository
git clone https://github.com/vishal-0122/DeepCare.git
cd DeepCare

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download data (from Google Drive link above)
# Place in data/ folder

# Run pipeline
python -m src.main configs/config.yaml
```

### 3️⃣ Docker Deployment

```bash
# Build & run with Docker Compose
docker-compose up --build

# Services will be available at:
# - FastAPI: http://localhost:8000
# - Streamlit UI: http://localhost:8501
# - API Docs: http://localhost:8000/docs
```

---

## 📖 Module Breakdown

| Module | Purpose |
|--------|---------|
| **main.py** | Orchestrates entire ML pipeline: load → preprocess → mine → balance → train → predict |
| **data_loader.py** | Loads CSV/XML data, merges demographics/visits/logs by patient_id |
| **preprocess.py** | Feature extraction from logs, categorical encoding, numerical scaling |
| **mining.py** | K-means clustering for patient cohort segmentation |
| **balancing.py** | Handles class imbalance via minority oversampling |
| **train.py** | Trains models with Bayesian hyperparameter optimization (Optuna) |
| **evaluate.py** | Computes precision, recall, F1, ROC-AUC metrics |
| **predict.py** | Generates batch/real-time dropout risk predictions |
| **config.py** | Parses YAML configuration files |
| **logger.py** | Structured logging for all pipeline stages |
| **tune.py** | Advanced hyperparameter tuning utilities |

---

## ⚙️ Configuration

Edit `configs/config.yaml` to customize:
- Data file paths
- Feature preprocessing rules
- Clustering parameters (k, algorithm)
- Model training settings (n_trials, feature columns)
- Evaluation metrics & output paths

```yaml
data:
  patient_demographics: 
    path: data/patient_demographics.csv
    key: "patient_id"
  target_column: "drop_off"

mining:
  algorithm: "kmeans"
  
training:
  n_trials: 30  # Bayesian optimization iterations
```

---

## 🔮 Workflow

1. **Data Loading** → Merge multi-source patient data
2. **Feature Engineering** → Extract 100+ features from hospital logs
3. **Preprocessing** → Encode categoricals, scale numericals, save scaler for deployment
4. **Mining** → Segment patients into K cohorts
5. **Balancing** → Oversample minority dropout class
6. **Training** → Hyperparameter tuning via Optuna, select best model
7. **Evaluation** → Compute metrics on test set
8. **Prediction** → Generate dropout risk scores for all patients
9. **Deployment** → Serve via FastAPI/Streamlit

---

## 📊 Exploratory Notebook

For full workflow walkthrough, analysis, and visualizations:
👉 Open `notebooks/deepcare.ipynb`

---

## 🔌 API Usage

### FastAPI Endpoint
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"patient_id": 12345, "features": [...]}'
```

### Streamlit UI
Navigate to `http://localhost:8501` for interactive prediction interface.

---

## 🐳 CI/CD Pipeline

This project uses **GitHub Actions** for automated testing and Docker image deployment:
- ✅ Automatic Docker image build & push on push to `main`
- ✅ Image tagged with `latest` and commit SHA
- ✅ Images available at Docker Hub: `docker pull vishal0122/deepcare:latest`

---

## 📋 Requirements

See `requirements.txt` for all dependencies. Key packages:

---

## 📝 License

This project is proprietary and contains sensitive healthcare data.

---

## 👨‍💻 Author

**Vishal Dangiwala**  
GitHub: [@vishal-0122](https://github.com/vishal-0122)

---

## 🤝 Support

For issues, questions, or contributions:
1. Check existing GitHub Issues
2. Open a new issue with detailed description