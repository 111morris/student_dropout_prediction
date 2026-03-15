# Student Dropout Risk Prediction System

**A Machine Learning Early Warning System for Secondary Schools**

This project provides a complete machine learning pipeline to predict the likelihood of student dropout using historic school data. The model is designed to act as an Early Warning System (EWS), enabling school administrators and teachers to intervene before a student officially drops out. 

The system was developed following a research-driven methodology, prioritizing **recall** (to minimize the number of at-risk students who go undetected) and integrating domain-specific feature engineering.

---

## Project Architecture

The pipeline is entirely automated and executes the following stages sequentially when run:

1. **Data Loading & Validation:** Loads raw data and checks for logical impossibilities (e.g., negative attendance, incorrect GPA ranges).
2. **Exploratory Data Analysis (EDA):** Automatically generates and saves 5 diagnostic visualization charts to the `reports/` directory.
3. **Data Cleaning:** Imputes missing values (median for numeric, mode for categorical) and automatically drops highly correlated features (multicollinearity check with threshold > 0.90) to prevent data leakage.
4. **Feature Engineering:** Creates derived behavioral and financial features (e.g., `GPA_CGPA_Diff`, `Study_Attendance_Ratio`, `Income_per_Travel`, `Overloaded_Flag`) and encodes categorical columns.
5. **Model Training & Cross-Validation:** Trains four models (Logistic Regression, Random Forest, Gradient Boosting, XGBoost) using Stratified 5-Fold Cross-Validation. Models are embedded in `sklearn.Pipeline` objects for clean scaling (e.g., only Logistic Regression uses `StandardScaler`). Class imbalance is handled natively via `class_weight="balanced"`.
6. **Threshold Tuning:** The best model's classification threshold is dynamically tuned to optimize for **Recall** rather than raw accuracy.
7. **Evaluation & Reporting:** Generates a comprehensive text report, plots ROC curves/confusion matrices, calculates SHAP/coefficient Feature Importance, and produces a final **Risk Score Dashboard** CSV for stakeholders.

---

## Project Structure

```text
student_dropout_prediction/
│
├── data/
│   ├── raw/                 # Contains source datasets (e.g., student_dropout_dataset_v3.csv)
│   └── processed/           # Contains generated cleaned_dataset.csv
│
├── docs/                    # Contextual research documents and references
│
├── models/                  # Saved serialized models (best_model.joblib, scaler.joblib)
│
├── reports/                 # Auto-generated outputs (Risk scores, EDA charts, Evaluation report)
│
├── src/                     # Source code modules
│   ├── __init__.py          
│   ├── utils.py             # Constants, config, and save/load utilities
│   ├── data_loader.py       # Loading and validation logic
│   ├── data_cleaner.py      # Imputation, deduplication, multicollinearity checking
│   ├── feature_engineer.py  # Derived features and encoding (One-Hot / Label)
│   ├── eda.py               # Generates distributions, correlation heatmaps, etc.
│   ├── model_trainer.py     # Training loops, CV, pipeline definition, threshold tuning
│   └── model_evaluator.py   # Grading, metrics, SHAP/importance extraction
│
├── main.py                  # The primary orchestrator script
├── requirements.txt         # Project dependencies
└── README.md                # This documentation file
```

---

## Current Model Performance (Benchmark)

On the primary dataset (`student_dropout_dataset_v3.csv`), the pipeline achieved the following results on the test set:

- **Best Model:** Logistic Regression
- **Baseline (Majority Class) Accuracy:** 76.45%
- **Model ROC-AUC:** 0.8201
- **Tuned Recall:** 0.8514 (Optimal threshold: 0.371)
- **Tuned Precision:** 0.3943

*Note: The model traded Precision for a significant boost in Recall (85.1%), which aligns perfectly with the research directive to minimize false negatives (missed dropouts).*

---

## Setup Instructions

You can run this project using either **Python `venv`** or **Conda**. All commands should be executed from the root directory (`student_dropout_prediction/`).

### Option A: Using Standard Python `venv` (Recommended for pure Python)

1. **Create the virtual environment**:
   ```bash
   python3 -m venv venv
   ```

2. **Activate the virtual environment**:
   - On Linux/macOS:
     ```bash
     source venv/bin/activate
     ```
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Option B: Using Anaconda / Miniconda (Recommended for Data Science)

1. **Create the Conda environment**:
   ```bash
   conda create --name dropout_env python=3.12 -y
   ```

2. **Activate the Conda environment**:
   ```bash
   conda activate dropout_env
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   *(Alternatively, you can install the packages using `conda install pandas numpy scikit-learn xgboost matplotlib seaborn joblib` but `pip` is guaranteed to match the exact `requirements.txt` file).*

---

## How to Run the Pipeline

Once your environment is set up and activated, you can run the entire pipeline with a single command:

```bash
python main.py
```

### What happens when you run it?
1. The script reads the raw data from `data/raw/student_dropout_dataset_v3.csv`.
2. It prints detailed log outputs to the terminal, tracking each stage (Validation → EDA → Cleaning → Feature Engineering → CV → Training → Evaluation).
3. It saves the transformed dataset to `data/processed/cleaned_dataset.csv`.
4. It exports the best model and scaler to the `models/` directory.
5. It yields all visualization charts and the text evaluation report to `reports/`.

---

## Interpreting the Outputs

After a successful run, navigate to the `reports/` folder. The most important files for schools and stakeholders are:

1. **`risk_scores.csv`**: A ready-to-use dashboard file. It lists every student in the test set alongside their predicted Risk Score (between 0.0 and 1.0) and translates this into a categorical Risk Level (🟢 Low, 🟡 Medium, 🔴 High).
2. **`evaluation_report.txt`**: A detailed breakdown of how each model performed, including Stratified 5-Fold Cross-Validation metrics and the final tuned classification report.
3. **`09_feature_importance.png`**: A bar chart visualizing the Top 10 factors that influence dropout risk (e.g., GPA, Stress Index, Attendance Rate). This provides *explainability* for why the model makes its decisions. 
4. **`02` to `07` image files**: Visual evidence of the pipeline's analysis (Correlation heatmaps, ROC curves, Confusion matrices).
