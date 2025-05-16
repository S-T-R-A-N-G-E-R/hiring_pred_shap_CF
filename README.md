# Candidate Hiring Predictor

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://candidatehiringpredictor.streamlit.app)
[![Python Version](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> Predict job candidate hiring outcomes with ML-powered insights and personalized improvement suggestions

## 📋 Table of Contents
- [Overview](#-overview)
- [Features](#-features)
- [Demo](#-demo)
- [Technology Stack](#-technology-stack)
- [Installation & Setup](#-installation--setup)
- [Usage & Examples](#-usage--examples)
- [Project Structure](#-project-structure)
- [Model Information](#-model-information)
- [Contributing](#-contributing)
- [Collaborators & Credits](#-collaborators--credits)
- [Contact](#-contact)

## 🔍 Overview

The Job Seeker Hiring Predictor is a machine learning-powered web application that predicts whether a job candidate will be hired based on their profile. Built using a Random Forest Classifier, the app not only provides hiring predictions but also explains rejections using SHAP (SHapley Additive exPlanations) and offers actionable suggestions to improve hiring chances using DiCE (Diverse Counterfactual Explanations).

The model achieves 93.3% accuracy on validation data and is trained on a dataset of 1,500 candidates with features like age, education level, experience, and interview scores.

This application is deployed on Streamlit Community Cloud, making it accessible to job seekers, recruiters, and HR professionals seeking data-driven insights into the hiring process.

## ✨ Features

- **Intelligent Prediction Engine** - Calculates hiring probability with Random Forest model
- **Explainable AI Integration** - Uses SHAP (SHapley Additive exPlanations) to identify key rejection factors
- **Actionable Improvement Suggestions** - Provides counterfactual scenarios via DiCE (Diverse Counterfactual Explanations)
- **User-Friendly Interface** - Clean two-column layout with intuitive form controls
- **Data Visualization** - Visual representation of feature importance and prediction confidence
- **Responsive Design** - Optimized for both desktop and mobile devices
- **Cloud Deployment** - Publicly accessible via Streamlit Cloud

## 🎬 Demo

> **Note**: Add your application screenshots to the repository and update the path below.
>
> Example: `![App Demo](screenshots/app_demo.png)`

You can try the live app on [Streamlit Cloud](https://candidatehiringpredictor.streamlit.app).

## 🛠️ Technology Stack

- **Frontend**: Streamlit with custom CSS
- **Backend**: Python 3.11+
- **Machine Learning**: scikit-learn, Random Forest Classifier
- **Explainable AI**: SHAP, DiCE
- **Data Processing**: pandas, NumPy
- **Visualization**: Matplotlib, Plotly
- **Deployment**: Streamlit Community Cloud

## 📦 Installation & Setup

### Prerequisites
- Python 3.11 or higher
- Git
- IDE (VS Code recommended)

### Local Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/S-T-R-A-N-G-E-R/hiring_pred_shap_CF.git
   cd hiring_pred_shap_CF
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run streamlit_job.py
   ```

The app will be available at http://localhost:8501

## 📊 Usage & Examples

### Basic Workflow

1. **Enter Candidate Profile Details**
   - Fill the form with your profile information
   - Include age, gender, education level, etc.

2. **Get Prediction Results**
   - Click "Predict Outcome" to see hiring probability
   - Review the key factors influencing the prediction

3. **Explore Improvement Suggestions**
   - Review counterfactual scenarios to improve outcomes
   - See how changing specific factors affects hiring probability

### Example Output

```
Prediction Result
Outcome: Not Hired
Probability of Being Hired: 0.060

Reason for Rejection
The feature most responsible is RecruitmentStrategy (reducing your hiring probability by 0.150).

Suggestions to Get Hired
Suggestion 1 (Predicted Hiring Probability: 0.930):
- Switch to Aggressive recruitment strategy

Suggestion 2 (Predicted Hiring Probability: 0.850):
- Upgrade to PhD degree (e.g., pursue a PhD)
```

## 📁 Project Structure

```
hiring_pred_shap_CF/
│
├── streamlit_job.py                              # Main Streamlit application
├── rf_classifier.joblib                          # Pre-trained Random Forest model
├── reference_data.joblib                         # Reference data for SHAP and DiCE
├── requirements.txt                              # Python dependencies
├── Final.ipynb                                   # Jupyter notebooks for model development
├── data.csv                                      # Original dataset files
├── shap_explanations_all_instances.csv           # SHAP explanation data
└── README.md                                     # Project documentation
```

## 🧠 Model Information

### Dataset
- 1,500 candidate profiles with hiring outcomes
- Features: age, gender, education level, experience, previous companies, distance from company, interview score, skill score, personality score, and recruitment strategy
- Target: Hiring outcome (binary: Hired/Not Hired)

### Model Performance
- **Algorithm**: Random Forest Classifier
- **Accuracy**: 93.3%
- **Features Used**: Education level, experience, interview scores, recruitment strategy, and more

## 👥 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 🙏 Collaborators & Credits

**Core Team:**
- [Swapnil Roy](https://github.com/S-T-R-A-N-G-E-R)
- Rupsha Das
- Nilanjan Dey

**Special Thanks:**
- [Streamlit](https://streamlit.io/) - For the amazing framework
- [scikit-learn](https://scikit-learn.org/) - For machine learning tools
- [SHAP](https://github.com/slundberg/shap) - For model explainability
- [DiCE](https://github.com/interpretml/DiCE) - For counterfactual explanations



Project Link: [https://github.com/S-T-R-A-N-G-E-R/hiring_pred_shap_CF](https://github.com/S-T-R-A-N-G-E-R/hiring_pred_shap_CF)

---

<p align="center">Last Updated: May 16, 2025</p>
