# 🔍 GTC-ML-Credit Card Fraud Detection 

## 📌 Project Overview  
Credit card fraud is one of the biggest threats in the financial sector, causing billions in annual losses and damaging customer trust.  
With the rise of **e-commerce and online transactions**, detecting fraudulent activity quickly and accurately is more important than ever.  

This project was developed as the **Final Project for the GTC-ML-Internship**.  
It aims to build a **fraud detection system** using the latest **2023 Credit Card Fraud Detection dataset** by analyzing transaction patterns and applying advanced Machine Learning and Deep Learning algorithms.  

---

## 👨‍💻 Team Members
- **Ibrahim Abdelsattar **  
- **Mohamed Abdelghany**  
- **Yousef Abdelhady**  
- **Yusuf Kamel**  
- **Mohamed Hamed**  
- **Omar Hosni**

---

## 🎯 Why This Project is Important  
- **Financial Security** – Preventing fraud reduces losses for both customers and banks.  
- **Customer Experience** – Accurate detection reduces unnecessary blocking of legitimate transactions.  
- **Evolving Fraud Patterns** – Fraud tactics change over time; updated datasets ensure adaptive detection methods.  
- **Balanced Dataset** – Unlike older versions, this dataset has a more balanced distribution of fraud vs. legitimate transactions, improving training quality.  

---

## 📂 Dataset Description  
The dataset contains anonymized **credit card transactions from 2023**, including both legitimate and fraudulent records.  

- **Amount** – Transaction amount (in local currency).  
- **TransactionType / Category** – Transaction type (purchase, transfer, withdrawal, etc.).  
- **V1, V2, … Vn** – Engineered/anonymized numerical features.  
- **Date / Time** – Transaction timestamp.  
- **Class** – Target variable:  
  - `0` → Legitimate transaction  
  - `1` → Fraudulent transaction  

📌 Source: [Kaggle – Credit Card Fraud Detection Dataset 2023]([https://www.kaggle.com/](https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023))  

---

## ⚙️ Project Workflow  

### 1. Data Understanding & Exploration (EDA)  
- Reviewed class distribution and balance.  
- Analyzed transaction amounts, time-based patterns, and correlations.  
- Visualized key patterns in fraud vs. legitimate transactions.  

### 2. Data Preprocessing  
- Handled missing values.  
- Normalized/scaled numerical features.  
- Encoded categorical variables.  
- Split data into training and testing sets.  

### 3. Model Building  
We implemented and compared the following algorithms:  
- Logistic Regression  
- Random Forest  
- XGBoost  
- LightGBM  
- CatBoost  
- Deep Neural Network (DNN)  

Cross-validation was applied to ensure robust evaluation.  

### 4. Model Evaluation  
Evaluation metrics:  
- **Accuracy**  
- **Precision**  
- **Recall**  
- **F1-Score**  
- **ROC-AUC**  

---

## 📊 Model Performance  

| Model                | Accuracy (%) |
|----------------------|--------------|
| Logistic Regression  | **82.19**    |
| Random Forest        | **86.45**    |
| XGBoost              | **99.74**    |
| LightGBM             | **95.41**    |
| CatBoost             | **97.35**    |
| Deep Neural Network  | **99.96**    |

---

## 📈 Observations  
- **Logistic Regression** performed the weakest → dataset too complex for linear models.  
- **Tree-based models (RF, LightGBM, CatBoost)** performed significantly better.  
- **XGBoost** excelled with **99.74% accuracy**, showing strong handling of feature complexity.  
- **DNN** achieved the highest accuracy (**99.96%**), proving best overall performance.  

---

## ✅ Conclusion  
Both **XGBoost** and **DNN** are highly effective for fraud detection.  
- **XGBoost** offers excellent performance with lower computational cost and higher interpretability.  
- **DNN** slightly outperforms XGBoost but at higher resource requirements.  

The choice between them depends on the **deployment environment** (real-time constraints vs. resource availability).  

---

## 🚀 Future Work  
- Improve explainability using **SHAP** or **LIME**.  
- Deploy models as a **REST API** for real-time fraud detection.  
- Explore ensemble approaches combining XGBoost & DNN.  
