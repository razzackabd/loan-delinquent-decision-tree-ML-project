# Loan Delinquency Prediction using Decision Tree

## 📌 Project Overview
This project aims to predict whether a loan applicant will become **delinquent (default)** or **non-delinquent** using a **Decision Tree Classifier**.  
The model helps financial institutions identify high-risk borrowers and reduce credit risk.

---

## 🎯 Objective
- Predict loan delinquency based on applicant and loan-related features
- Improve loan approval decisions
- Reduce financial loss for banks and lending institutions

---

## 📂 Dataset
The dataset contains information related to:
- Applicant details (age)
- Credit information (credit score, previous defaults)

**Target Variable:**
- `Loan_Status`  
  - 1 → Delinquent  
  - 0 → Non-Delinquent

---

## ⚙️ Technologies Used
- Python
- Jupyter Notebook
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn

---

## 🧹 Data Preprocessing
- Handled missing values
- Encoded categorical variables using Label Encoding / One-Hot Encoding
- Split the dataset into training and testing sets
- No feature scaling required (Decision Tree is scale-independent)

---

## 🤖 Model Used
- **Decision Tree Classifier**
- Hyperparameters tuned:
  - `max_depth`
  - `min_samples_split`
  - `min_samples_leaf`

**Why Decision Tree?**
- Easy to understand and interpret
- Handles both numerical and categorical data
- Captures non-linear relationships

---

## 📊 Model Evaluation
The model performance was evaluated using:
- Confusion Matrix
- Accuracy Score
- Precision, Recall, F1-score
- ROC-AUC Score (optional)

**Result:**
- The model achieved good accuracy and was able to identify delinquent customers effectively.

---

## 🔍 Feature Importance
Important factors influencing loan delinquency:
- Credit Score
- Loan Amount
- Applicant Income

Decision Tree helps visualize how these features affect the final prediction.

---

## 📈 Results & Conclusion
- The model successfully predicts loan delinquency
- Helps banks reduce financial risk
- Improves decision-making during loan approval

---

## 🚀 Future Improvements
- Use ensemble models like Random Forest or XGBoost
- Perform hyperparameter tuning using GridSearchCV
- Handle class imbalance using SMOTE
- Deploy the model using Flask or Streamlit

--
# loan-delinquent-decision-tree-ML-project
