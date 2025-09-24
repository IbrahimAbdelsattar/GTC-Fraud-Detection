# 🔍 Advanced Credit Card Fraud Detection using a Stacking Ensemble

## 🚀 Live Demo
Experience the model in action by testing real-time or batch transactions on our interactive web app:

**➡️ [Hugging Face Spaces: Fraud Detection Demo](https://huggingface.co/spaces/useifabdelhady/FraudDetection)**      
![Fraud Detection app](https://github.com/user-attachments/assets/0b07a4b1-7a94-44b1-9260-abe272220dc5)

---

## 📌 Project Overview
Credit card fraud poses a significant threat to the financial industry, leading to substantial annual losses and eroding consumer trust. The primary challenge in detecting fraud lies in identifying rare fraudulent transactions hidden within millions of legitimate ones.

This project, developed as the **Final Project for the GTC-ML-Internship**, tackles this challenge by building a sophisticated fraud detection system. It leverages an advanced **Stacking Ensemble model** combined with extensive **feature engineering** and techniques to handle **extreme class imbalance**. The model is trained on the classic credit card fraud dataset, which features anonymized transaction data.

---

## 👨‍💻 Team Members
- **Ibrahim Abdelsattar**
- **Mohamed Abdelghany**
- **Yousef Abdelhady**
- **Yusuf Kamel**
- **Mohamed Hamed**
- **Omar Hosni**

---

## 📂 Dataset Description
The project utilizes the highly imbalanced **Credit Card Fraud Detection** dataset from Kaggle. It contains anonymized transactions made by European cardholders.

- **Features**: The dataset consists of 30 numerical features.
  - `Time` & `Amount`: The only non-anonymized features.
  - `V1` to `V28`: Anonymized features resulting from a PCA transformation.
- **Class Imbalance**: The dataset is extremely imbalanced, with fraudulent transactions accounting for only **0.17%** of all records. This makes accuracy a poor metric and requires specialized techniques.
- **Target Variable (`Class`)**:
  - `0` → Legitimate transaction
  - `1` → Fraudulent transaction

📌 **Source**: [Kaggle – Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

---

## ⚙️ Project Workflow & Methodology

### 1. Exploratory Data Analysis (EDA)
- Analyzed the severe class imbalance and its implications.
- Visualized distributions of `Time` and `Amount` for both classes.
- Investigated correlations between features to identify key predictors of fraud.

### 2. Advanced Feature Engineering
To enhance the model's predictive power, several new features were created:
- **Temporal Features**: `hour_of_day` and `time_bin` were extracted from the `Time` feature.
- **Amount-Based Features**: `scaled_amount`, `amount_deviation` (from the mean), and `amount_bin` were created to capture spending patterns.
- **Aggregated Features**: `mean_V` and `std_V` were calculated from the PCA components (V1-V28).
- **Interaction Features**: Top correlating features were combined to capture complex, non-linear relationships.

### 3. Model Architecture: Stacking Ensemble
A powerful **Stacking Classifier** was built to combine the strengths of multiple high-performing gradient boosting models.
- **Base Models**:
  - **XGBoost**
  - **CatBoost**
  - **LightGBM**
- **Meta-Model**:
  - **Logistic Regression** was used as the final estimator to aggregate the predictions from the base models.

### 4. Handling Class Imbalance
- **SMOTE (Synthetic Minority Over-sampling Technique)** was integrated into an `ImbPipeline`. This technique generates synthetic samples for the minority class (fraud) to create a more balanced training set, preventing the model from being biased towards the majority class.

### 5. Evaluation & Threshold Optimization
- The model was validated using **5-fold stratified cross-validation**.
- Given the imbalance, the focus was on metrics like **Precision, Recall, F1-Score, and PR-AUC**.
- A critical step was **optimizing the decision threshold**. Instead of the default 0.5, we identified the optimal threshold (`0.9821`) that maximizes the F-beta score (with `beta=2.0`), which heavily prioritizes **Recall** (catching as many fraudulent transactions as possible).

---

## 📊 Final Model Performance
The table below shows the performance of the final **Stacking Ensemble + SMOTE** model on the test set using the optimized decision threshold.

| Metric | Score |
| :--- | :--- |
| **Accuracy** | 99.96% |
| **Precision** | 92.31% |
| **Recall** | 85.71% |
| **F1-Score** | **0.8889** |
| **ROC-AUC** | 0.9847 |
| **PR-AUC** | 0.8711 |


![The Confusion Matrix](https://github.com/user-attachments/assets/24f9245f-4fab-4e46-bd4c-fb29319f7998)

---

## ✅ Conclusion
This project successfully demonstrates that a combination of deep feature engineering, a powerful stacking ensemble, and specialized techniques like SMOTE and threshold optimization can build a highly effective fraud detection system. The final model achieves an excellent **F1-Score of 0.89**, successfully balancing the need to catch fraudulent transactions (high recall) while minimizing false alarms (high precision).

