import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, 
                              AdaBoostClassifier, BaggingClassifier)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb

# Import the evaluation function from model_evaluation.py
from src.model_evaluation import evaluate_model

def train_and_evaluate_models(X, y):
    """
    Splits the dataset into training and testing sets, trains multiple machine learning models,
    and evaluates each model using the evaluate_model() function. The evaluation includes metrics
    such as accuracy, precision, recall, F1-score, and AUC-ROC, along with plots for the confusion 
    matrix and the ROC curve.

    Parameters:
        X (DataFrame or ndarray): Feature set.
        y (Series or ndarray): Target variable.
    
    Returns:
        results (dict): A dictionary containing accuracy (and optionally, other metrics) for each model.
    """
    # Split data into training and testing sets (80/20 split)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define a dictionary of models to evaluate
    models = {
        "Logistic Regression": LogisticRegression(max_iter=5000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVC": SVC(random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=42),
        "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Naive Bayes": GaussianNB(),
        "Bagging - Decision Tree": BaggingClassifier(
            base_estimator=DecisionTreeClassifier(random_state=42),
            n_estimators=50,
            random_state=42
        )
    }
    
    results = {}
    
    # Iterate over the models dictionary, train and evaluate each model
    for model_name, model in models.items():
        print(f"\nTraining and evaluating model: {model_name}")
        model.fit(X_train, y_train)
        
        # Predict on test set
        y_pred = model.predict(X_test)
        
        # Get probability estimates for ROC curve - either via predict_proba or decision_function
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = model.decision_function(X_test)
        
        # Use the evaluation function to print metrics and plot confusion matrix and ROC curve
        evaluate_model(y_test, y_pred, y_prob, model_name=model_name)
        
        # Store accuracy in results dictionary (you can extend this with additional metrics)
        accuracy = np.mean(y_pred == y_test)
        results[model_name] = {"accuracy": accuracy}
        
    return results