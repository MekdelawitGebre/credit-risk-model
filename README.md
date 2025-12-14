#  Credit Risk Probability Model for Alternative Data

### An End-to-End Implementation for Building, Deploying, and Automating a Credit Risk Model

---

##  1. Overview

**Organization:** Bati Bank  
**Objective:** Build a credit scoring model using alternative eCommerce behavioral data.  

Bati Bank is partnering with a leading eCommerce company to launch a **Buy-Now-Pay-Later (BNPL)** service that allows customers to make purchases on credit.  
The goal is to create a **Credit Risk Probability Model** that uses customer transactional behavior to estimate the likelihood of default.

This project follows a complete **MLOps workflow**:
- Data ingestion and cleaning  
- Exploratory Data Analysis (EDA)  
- Feature Engineering & Proxy Target Creation  
- Model training and evaluation  
- Experiment tracking (MLflow)  
- API deployment (FastAPI + Docker)  
- Continuous Integration (GitHub Actions)  

---

##  2. Repository Structure

credit-risk-model/
├── .github/workflows/ci.yml # CI/CD workflow for linting & testing
├── data/
│ ├── raw/ # Raw data files (add to .gitignore)
│ └── processed/ # Cleaned and processed data for training
├── notebooks/
│ └── eda.ipynb # Exploratory Data Analysis notebook
├── src/
│ ├── init.py
│ ├── data_processing.py # Feature engineering & RFM calculations
│ ├── train.py # Model training with MLflow logging
│ ├── predict.py # Model inference script
│ └── api/
│ ├── main.py # FastAPI REST API for deployment
│ └── pydantic_models.py # Pydantic models for data validation
├── tests/
│ └── test_data_processing.py # Unit tests for data processing
├── Dockerfile # Container setup for deployment
├── docker-compose.yml # Container orchestration
├── requirements.txt # Dependencies
└── README.md # Project documentation


---

##  3. Credit Scoring Business Understanding

### **3.1 Basel II and the Need for Interpretability**

The **Basel II Capital Accord** defines regulatory standards for managing credit risk.  
It requires financial institutions to use **transparent, interpretable, and auditable** models to assess borrowers' creditworthiness.

This means:
- Model assumptions and logic must be explainable.  
- Risk measurement must be traceable and well-documented.  
- Decisions must be defensible to regulators and stakeholders.

Hence, our model must prioritize **interpretability** alongside predictive performance.

---

### **3.2 Why We Need a Proxy Target Variable**

The dataset lacks an explicit “default” or “credit risk” label.  
To overcome this, we engineer a **proxy variable (`is_high_risk`)** that categorizes users based on behavioral data:

- Customers with **low spending frequency** and **low monetary value** are considered *high risk*.  
- Customers with **high frequency** and **high monetary value** are *low risk*.  

This proxy helps simulate a real-world risk prediction scenario.  
However, it introduces **business risks** such as:
- Potential misclassification of customers.  
- Overgeneralization from behavior to creditworthiness.  
- Bias from country, product, or channel patterns.

---

### **3.3 Model Trade-offs: Interpretability vs. Performance**

| Model Type | Advantages | Limitations |
|-------------|-------------|-------------|
| **Logistic Regression (with Weight of Evidence)** | Interpretable, Basel-compliant, simple to document | May miss nonlinear relationships |
| **Random Forest / Gradient Boosting (e.g., XGBoost)** | Strong predictive power, handles nonlinearity | Harder to interpret, needs explainability tools (SHAP/LIME) |

In regulated industries, **interpretability and auditability** often outweigh minor accuracy improvements.

---

##  4. Data Description

| Column Name | Definition |
|--------------|-------------|
| TransactionId | Unique transaction identifier |
| BatchId | Unique batch identifier for grouped transactions |
| AccountId | Customer’s account number |
| SubscriptionId | Subscription linked to the account |
| CustomerId | Unique customer identifier |
| CurrencyCode | Transaction currency (e.g., UGX) |
| CountryCode | Numeric country code |
| ProviderId | Source provider of the purchased item |
| ProductId | Item being purchased |
| ProductCategory | Category grouping of the product |
| ChannelId | Platform used (web, Android, iOS, etc.) |
| Amount | Transaction amount (positive = debit, negative = credit) |
| Value | Absolute value of the transaction amount |
| TransactionStartTime | Timestamp of the transaction |
| PricingStrategy | Pricing model for the merchant |
| FraudResult | Fraud indicator (1 = Fraudulent, 0 = Legitimate) |

---

##  5. Project Workflow

### **Task 1 – Business Understanding**
- Review Basel II Accord and Credit Scoring Principles.
- Document business understanding in README .

### **Task 2 – Exploratory Data Analysis (EDA)**
- Analyze data structure and distribution.  
- Identify missing values and outliers.  
- Visualize key numerical and categorical patterns.  
- Summarize top 3–5 insights.  

### **Task 3 – Feature Engineering**
- Create RFM metrics (Recency, Frequency, Monetary).  
- Aggregate transaction features per customer.  
- Encode categorical variables.  
- Handle missing values and scaling.  
- Compute Weight of Evidence (WoE) and Information Value (IV).  

### **Task 4 – Proxy Target Creation**
- Segment customers using K-Means clustering.  
- Label least engaged cluster as `is_high_risk = 1`.  
- Merge back with processed dataset for model training.  

### **Task 5 – Model Training and Tracking**
- Train Logistic Regression and Gradient Boosting models.  
- Tune hyperparameters with GridSearchCV.  
- Track experiments using **MLflow**.  
- Evaluate using Accuracy, Precision, Recall, F1, ROC-AUC.  
- Register best model in MLflow Model Registry.  

### **Task 6 – Deployment and CI/CD**
- Build a REST API using **FastAPI** for real-time predictions.  
- Containerize the model with **Docker** and **docker-compose**.  
- Set up **GitHub Actions** for:
  - Linting (flake8)
  - Unit tests (pytest)
  - Automated build validation

---

##  6. Key Learning Outcomes

**Skills Gained:**
- Data preprocessing and feature engineering  
- Model building, evaluation, and hyperparameter tuning  
- MLflow experiment tracking and model registry  
- API development (FastAPI)  
- Docker-based deployment  
- Continuous Integration & Testing (GitHub Actions)  

**Knowledge Gained:**
- Credit risk modeling principles  
- RFM analysis for alternative data  
- Basel II regulatory compliance in model design  
- Trade-offs between interpretability and accuracy  

---

##  7. Tools & Technologies

| Category | Tools |
|-----------|--------|
| Programming | Python 3.11 |
| Data Analysis | pandas, numpy, seaborn, matplotlib |
| Machine Learning | scikit-learn, XGBoost, LightGBM |
| Experiment Tracking | MLflow |
| API Development | FastAPI, Pydantic |
| Deployment | Docker, docker-compose |
| Testing & CI/CD | pytest, flake8, GitHub Actions |

---

##  8. How to Run Locally

```bash
# Clone the repository
git clone https://github.com/MekdelawitGebre/credit-risk-model.git
cd credit-risk-model

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate (on Windows)

# Install dependencies
pip install -r requirements.txt

# Run FastAPI server
uvicorn src.api.main:app --reload

The API will be available at:

http://127.0.0.1:8000/docs