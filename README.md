# 🚗 A3: Car Price Prediction (Classification)

### 👨‍💻 By Lu Htoo Kyaw (ST124956)

🔗 **Live Website**: [https://st124956.ml.brain.cs.ait.ac.th/](https://st124956.ml.brain.cs.ait.ac.th/)

This project tackles car price prediction as a **classification problem** using **Multinomial Logistic Regression**, extending from previous regression-based implementations. It includes a full pipeline from data preprocessing and modeling to deployment using Docker and MLflow, with CI/CD integration through GitHub Actions.

---

## 📦 Features

- 🔢 Converts `selling_price` into 4 discrete classes (0–3) using bucketing.
- 🧮 Custom implementation of **Logistic Regression** with:
  - Accuracy, Precision, Recall, F1-score (per class)
  - Macro and Weighted metrics
- 🧪 Comparison with Scikit-learn’s `classification_report`
- 🧰 Optional **Ridge (L2) Regularization**
- 📈 Tracked training with **MLflow** (remote server)
- 🚀 Deployed Django web application
- ✅ CI/CD with **GitHub Actions**
- 🐳 Containerized via **Docker Compose**

---

## 🧠 Model Details

The model transforms the car price prediction into a 4-class classification task by dividing the prices into bins. A custom logistic regression model was implemented with optional Ridge regularization:


Manual implementation of classification metrics:
- **Accuracy**
- **Precision, Recall, F1-score (per class)**
- **Macro averaging**
- **Weighted averaging**

---

## 🛠️ Installation & Usage

### 🔧 1. Clone the Repository

```bash
git clone https://github.com/luhtookyaw/a3-car-price-prediction.git
cd a3-car-price-prediction
```

### 🐍 2. Install Python Packages (Local)
```bash
cd app
pip3 install -r requirements.txt
python3 manage.py runserver
```
### 🐳 Docker Deployment

Build from Scratch
```bash
cd app
docker compose up -d
```
Pull and Run (Prebuilt)
```bash
docker compose up -d
```
### 📊 MLflow Integration
* Tracking URI: https://mlflow.ml.brain.cs.ait.ac.th/
* Experiment Name: st124956-a3
* Model Name: st124956-a3-model
* Stage: Staging

Experiments are logged with:
* Accuracy and F1 scores
* Hyperparameter settings
* Regularization choice
* Trained models and metrics

### 🔁 CI/CD Pipeline
#### CI/CD Overview:
* Unit Tests:
  * ✅ Checks if the model accepts valid input
  * ✅ Verifies the prediction output shape

* GitHub Actions:
  * Runs tests on every push
  * Deploys the application automatically if tests pass



