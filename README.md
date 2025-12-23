# 🛡️ Network Anomaly Detection System

An AI-powered network intrusion detection system using machine learning to classify network traffic as Normal or specific attack types (DoS, Probe, R2L, U2R).

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![ML](https://img.shields.io/badge/ML-RandomForest-green.svg)
![Accuracy](https://img.shields.io/badge/Accuracy-74.5%25-orange.svg)
![License](https://img.shields.io/badge/License-MIT-purple.svg)

## 🎯 Features

- **Multi-Class Attack Detection**: Identifies 5 traffic categories
  - ✅ Normal Traffic
  - 💥 DoS (Denial of Service)
  - 🔍 Probe (Reconnaissance)
  - 🔓 R2L (Remote to Local)
  - 👤 U2R (User to Root)
  
- **Real-Time Dashboard**: Beautiful web interface with live statistics
- **25+ Vulnerability Indicators**: CVE references, MITRE ATT&CK mappings
- **Explainable AI**: Shows top contributing factors for each prediction
- **REST API**: Easy integration with existing security tools

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| Overall Accuracy | 74.5% |
| Precision | ~75% |
| Recall | ~75% |
| F1-Score | ~75% |

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/Baasil22/Network-Anomaly-Detection-System.git
cd Network-Anomaly-Detection-System
```

### 2. Create Virtual Environment
```bash
python -m venv env
# Windows
env\Scripts\activate
# Linux/Mac
source env/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Dataset
Download NSL-KDD dataset files to `data/raw/`:
- [KDDTrain+.txt](https://www.unb.ca/cic/datasets/nsl.html)
- [KDDTest+.txt](https://www.unb.ca/cic/datasets/nsl.html)

### 5. Train the Model
```bash
python train.py
```

### 6. Run the Application
```bash
python api/app.py
```

### 7. Access Dashboard
Open http://localhost:5000 in your browser.

## 📁 Project Structure

```
Network-Anomaly-Detection/
├── api/                    # Flask API
│   ├── app.py             # Main API server
│   ├── predictor.py       # ML prediction service
│   └── detection_engine.py # Rule-based enhancement
├── dashboard/              # Web Interface
│   ├── index.html         # Main dashboard
│   ├── styles.css         # Sunset theme styling
│   └── script.js          # Real-time updates
├── data/                   # Dataset directory
│   ├── raw/               # NSL-KDD files (download)
│   └── download_data.py   # Dataset downloader
├── models/                 # Saved models
│   └── saved/             # Trained model files
├── src/                    # Core ML code
│   ├── config.py          # Configuration
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── evaluation.py
│   └── models/            # Model architectures
├── train.py               # Training script
├── requirements.txt       # Dependencies
└── README.md
```

## 🔌 API Endpoints

### Health Check
```http
GET /health
```

### Predict Single Sample
```http
POST /api/predict
Content-Type: application/json

{
  "features": [0, "tcp", "http", "SF", 215, 45076, ...]
}
```

### Response
```json
{
  "label": "Normal",
  "threat_level": "safe",
  "confidence": 0.98,
  "action": "ALLOW",
  "attack_type": null,
  "explanation": {...},
  "top_factors": [...]
}
```

## 🛡️ Attack Types Detected

| Type | Description | Severity |
|------|-------------|----------|
| DoS | Denial of Service (SYN flood, Smurf, Neptune) | CRITICAL |
| Probe | Reconnaissance (port scan, Nmap, Satan) | MEDIUM |
| R2L | Remote to Local (brute force, password guessing) | HIGH |
| U2R | User to Root (privilege escalation, rootkit) | CRITICAL |

## 📈 Dataset

Uses the **NSL-KDD** dataset, an improved version of KDD Cup 1999:
- Training samples: ~125,000
- Test samples: ~22,500
- 41 network features + attack label

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License.

## 👨‍💻 Author

**Baasil** - [GitHub](https://github.com/Baasil22)

---

⭐ Star this repo if you find it useful!
