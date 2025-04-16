# Hotel Booking Cancellation Prediction  
**Special Topics in Data Science (DS290)**  
**Author:** Eduard Petrosyan  
**Date:** March 26, 2025  

## 📌 Project Overview  
This project aims to predict hotel booking cancellations using a variety of machine learning models. The target variable is `is_canceled` (1 if canceled, 0 otherwise). Accurate prediction allows hotels to improve revenue management strategies like deposit policies and dynamic pricing.

---
## 📊 Data Description  
The dataset includes records from a City Hotel and a Resort Hotel with features such as:  
- `is_canceled`, `lead_time`, `adr`, `arrival_date`, `stay_duration`, `guest_details`  
- Booking history, distribution channel, customer type, deposit type, etc.

Irrelevant or sensitive columns like name, email, phone, and credit card info were removed.

---

## 🧹 Data Preprocessing  
- Dropped irrelevant columns (`name`, `email`, etc.)
- Filled missing values in `children` with 0  
- Dropped high-cardinality columns (`agent`, `company`)  
- Engineered new features: `total_stay_nights`, `total_guests`  
- Applied one-hot encoding to categorical variables  
- Removed ADR outliers (`adr > 5000`)  
- Final dataset saved to `data/processed/processed_hotel_booking.csv`

---

## 📈 Exploratory Data Analysis (EDA)  
Performed using histograms, boxplots, and heatmaps to explore:  
- Distributions  
- Outliers  
- Correlations  
- Categorical class balance  

---

## 🤖 Machine Learning Models  
The following models were trained and evaluated:  
- Logistic Regression  
- Decision Tree  
- Random Forest  
- Gradient Boosting, AdaBoost, XGBoost  
- K-Nearest Neighbors (KNN), Naive Bayes  
- Bagging (with Decision Trees)

Metrics used: **Accuracy, Precision, Recall, F1-Score, AUC-ROC**  
Evaluation is visualized with confusion matrices and ROC curves.

---

## 🔧 Hyperparameter Tuning  
Tuning was performed for Random Forest using `GridSearchCV`.  
Best parameters:
```python
{
  'bootstrap': False,
  'max_depth': 30,
  'max_features': 'sqrt',
  'min_samples_leaf': 2,
  'min_samples_split': 2,
  'n_estimators': 100
}
```
---
## ✅ Conclusion
After running multiple models and tuning, Random Forest showed the best performance. While results are promising, they are not perfect — likely due to feature limitations. Future improvements may include better feature engineering or incorporating external factors.
