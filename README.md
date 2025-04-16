Special Topics in Data Science (DS290) – Hotel Booking Cancellation Prediction
Project Title: Hotel Booking Cancellation Prediction
Date: March 26, 2025
Author: Eduard Petrosyan

Project Overview
This project focuses on predicting hotel booking cancellations using a variety of machine learning models. The primary target variable is is_canceled, where the goal is to determine whether a booking will be canceled. By accurately predicting cancellations, hotels can implement proactive revenue management strategies (such as requiring deposits or adjusting pricing policies) to mitigate revenue loss.

Project Structure
The project is organized as follows:

nginx
Copy
Project Root/
├── data/
│   ├── raw/
│   │   └── hotel_booking.csv          # Original dataset containing booking information
│   └── processed/
│       └── processed_hotel_booking.csv  # Cleaned and preprocessed dataset
├── notebooks/
│   └── Notebook.ipynb                 # Jupyter Notebook containing EDA and model application
├── src/
│   ├── model_evaluation.py            # Module with functions to evaluate model performance
│   └── model_training.py              # Module to train and compare different ML models
└── README.md                          # Project documentation and instructions
Data Description
The dataset consists of booking records from two hotels (a City Hotel and a Resort Hotel). Key features include:

is_canceled: Binary indicator (1 if canceled, 0 otherwise).

lead_time: Number of days between booking entry and arrival.

Arrival information: Year, week number, and day of month of arrival.

Stay duration: Number of weekend and week nights booked.

Guest details: Numbers of adults, children, and babies.

Booking history: Previous cancellations, non-canceled bookings, and booking changes.

ADR (Average Daily Rate): Per-night revenue for a booking.

Additional details: Customer type, distribution channel, deposit type, etc.

Some columns (e.g., name, email, phone number, credit card) were considered irrelevant and removed during preprocessing.

Data Preprocessing
The preprocessing steps include:

Cleaning the Data:

Dropping unneeded columns such as name, email, phone-number, credit_card, reservation_status, and reservation_status_date.

Handling missing values:

Replace missing values in the children column with 0.

Drop highly unique columns like agent and company.

Feature Engineering:

Creating new features like total_stay_nights (sum of week and weekend nights) and total_guests (sum of adults, children, and babies).

Encoding:

One-hot encoding is applied to categorical features (e.g., hotel, arrival_date_month, distribution_channel, deposit_type, customer_type).

Outlier Removal:

Removing outlier rows where adr is higher than 5000.

Data Saving:

The cleaned dataset is saved as data/processed/processed_hotel_booking.csv.

Example code to save the processed data:

python
Copy
output_path = "../data/processed/processed_hotel_booking.csv"
df.to_csv(output_path, index=False)
print(f"Cleaned data saved to {output_path}")
Exploratory Data Analysis (EDA)
EDA is performed to understand the data distributions, identify outliers, and explore relationships among features using statistical summaries and visualization tools like histograms, boxplots, and correlation heatmaps.

For example, histograms and boxplots are used to inspect the numerical features, while count plots are used for categorical features.

Model Training and Evaluation
Several machine learning algorithms are applied to the dataset, including:

Logistic Regression

Decision Tree

Random Forest

Gradient Boosting

AdaBoost

XGBoost

K-Nearest Neighbors (KNN)

Naive Bayes

Bagging (with Decision Tree as base estimator)

Each model is evaluated using metrics such as Accuracy, Precision, Recall, F1-Score, and AUC-ROC. Custom evaluation functions in src/model_evaluation.py generate confusion matrices and ROC curves for visual assessment.

For example, the evaluation function is called as follows:

python
Copy
evaluate_model(y_test, y_pred, y_prob, model_name="Model Name")
Hyperparameter Tuning
Hyperparameter tuning is performed (using GridSearchCV) on the selected model (Random Forest) to optimize its performance. An expanded grid of hyperparameters is tested using 5-fold cross-validation.
The best parameters found were:

python
Copy
{'bootstrap': False, 'max_depth': 30, 'max_features': 'sqrt', 'min_samples_leaf': 2, 'min_samples_split': 2, 'n_estimators': 100}
Example snippet for hyperparameter tuning:

python
Copy
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from src.model_evaluation import evaluate_model

# Load data and split
df = pd.read_csv("../data/processed/processed_hotel_booking.csv")
X = df.drop("is_canceled", axis=1)
y = df["is_canceled"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid = {
    'n_estimators': [100, 200, 300, 400],
    'max_depth': [10, 20, 30, 40],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [2, 4, 6],
    'max_features': ['sqrt', 'log2'],
    'bootstrap': [True, False]
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)
print("Best Hyperparameters:", grid_search.best_params_)
best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(X_test)
if hasattr(best_rf, "predict_proba"):
    y_prob = best_rf.predict_proba(X_test)[:, 1]
else:
    y_prob = best_rf.decision_function(X_test)
evaluate_model(y_test, y_pred, y_prob, model_name="Tuned Random Forest")
Running the Models
In the notebook located in the notebooks/ folder, import your model training functions from src/model_training.py, load the processed data, and call the functions to train and evaluate multiple models.

Example notebook usage:

python
Copy
import sys
sys.path.append("..")  # Adjust PYTHONPATH to access the src folder
from src.model_training import train_and_evaluate_models
import pandas as pd

df_encoded = pd.read_csv("../data/processed/processed_hotel_booking.csv")
X = df_encoded.drop("is_canceled", axis=1)
y = df_encoded["is_canceled"]
model_results = train_and_evaluate_models(X, y)

print("\nComparative Accuracy Results:")
for model_name, metrics in model_results.items():
    print(f"{model_name}: Accuracy = {metrics['accuracy']:.4f}")
Conclusion
After conducting EDA, applying various machine learning algorithms, and tuning model hyperparameters, our evaluation indicates that Random Forest (both untuned and tuned) performs best for predicting cancellations. The tuned model shows marginal improvements in key metrics, making it a strong candidate for deployment to assist in hotel revenue management decisions.