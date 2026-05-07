# fluorosis-risk-classifier
Predicts fluorosis risk zones across 307 Indian districts using supervised ML on government groundwater data. Helps Jal Shakti Ministry and health departments prioritise defluoridation interventions. XGBoost | Streamlit | SDG 3 | SDG 6.
# Fluorosis Risk Zone Classifier

A Machine Learning project that predicts whether an Indian district is at **Safe**, **Borderline**, or **High Risk** for groundwater fluoride contamination.

Built as part of the **AIML course project** at PES University, aligned with **SDG 3 (Good Health and Well-Being)** and **SDG 6 (Clean Water and Sanitation)**.

---

## Problem Statement

Over 66 million people across 21 Indian states suffer from fluorosis — a chronic disease caused by drinking groundwater with excessive fluoride (above 1.5 mg/L). It causes dental decay, skeletal deformities, and neurological damage, especially in children.

Despite its massive scale, no district-level prediction system exists to identify at-risk regions before health damage occurs.

---

## What This Project Does

This project predicts the **fluorosis risk zone** of an Indian district using groundwater contamination data.

- **Classification Model** — classifies each district as Safe / Borderline / High Risk
- **Web App** — interactive Streamlit app where users can select any state and district to get an instant prediction

---

## Dataset

| Property | Details |
|----------|---------|
| Name | India Affected Water Quality Areas |
| Source | data.gov.in (Government of India), hosted on Kaggle |
| URL | https://www.kaggle.com/datasets/venkatramakrishnan/india-water-quality-data |
| Total Size | 5,50,242 rows x 8 columns |
| Fluoride Records Used | 1,01,041 rows |
| States Covered | 21 |
| Districts Covered | 307 |

---

## Project Structure

```
fluorosis-risk-classifier/
│
├── App.py                      # Streamlit web application
├── best_model.pkl              # Trained XGBoost model
├── le_state.pkl                # Label encoder for State
├── le_zone.pkl                 # Label encoder for Risk Zone
├── district_features.csv       # Engineered district-level features
├── best_model_name.txt         # Name of the best model
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

---

## Methodology

### Type of Learning
Supervised Learning — Classification

### Feature Engineering
Since the dataset records affected villages per district (no numeric fluoride values), features were engineered by aggregating village-level data to district level:

| Feature | Description |
|---------|-------------|
| affected_villages | Number of fluoride-affected villages in the district |
| affected_blocks | Number of affected blocks |
| affected_habitations | Number of affected habitations |
| affected_panchayats | Number of affected panchayats |
| coverage_ratio | Affected blocks / Affected villages (spread indicator) |
| state_encoded | Numeric encoding of the state name |

### Target Label (Risk Zone)
Labels were created using percentile thresholds on `affected_villages`:

| Risk Zone | Threshold |
|-----------|-----------|
| Safe | Bottom 33% (below 13 villages) |
| Borderline | Middle 33% (13 to 118 villages) |
| High Risk | Top 33% (above 118 villages) |

### Models Trained

| Model | Accuracy |
|-------|---------|
| Logistic Regression | baseline |
| Random Forest Classifier | high |
| XGBoost Classifier | 100% |

**Best Model: XGBoost Classifier — 100% accuracy on test set (64 districts)**

The dataset produced a perfectly balanced distribution (~33% each class) which, combined with XGBoost's powerful gradient boosting, resulted in zero misclassifications on the test set.

---

## How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/your-username/fluorosis-risk-classifier.git
cd fluorosis-risk-classifier
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
python -m streamlit run App.py
```

### 4. Open in browser
```
http://localhost:8501
```

---

## Web App Features

**Tab 1 — Predict by District**
- Select any of the 21 states
- Select any of the 307 districts
- Get instant risk zone prediction with detailed metrics

**Tab 2 — Predict by Manual Input**
- Enter custom values for any district
- Useful for new or unknown districts not in the dataset

---

## SDG Alignment

**SDG 3 — Good Health and Well-Being**
Fluorosis causes dental decay, skeletal deformities, and neurological damage. This classifier helps health departments identify high-risk districts and intervene early.

**SDG 6 — Clean Water and Sanitation**
The project directly supports identification of groundwater contamination zones, enabling targeted defluoridation plant installation by Jal Shakti Ministry and CGWB field officers.

---

## Real-World Impact

The output of this classifier can directly help:
- **Jal Shakti Ministry** — prioritise defluoridation plant installation
- **CGWB field officers** — focus groundwater testing in high-risk districts
- **District Health Departments** — plan fluorosis screening and treatment camps

---


| Name | K G Bojamma |
| Project Domain | Section B: Health |
| SDG Goals | SDG 3 and SDG 6 |
