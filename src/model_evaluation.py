import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score, roc_curve)

def evaluate_model(y_true, y_pred, y_prob, model_name="Model"):
    """
    Evaluate a classification model with key metrics and plots.
    
    Parameters:
        y_true (array-like): Ground truth (correct) target values.
        y_pred (array-like): Estimated targets as returned by a classifier.
        y_prob (array-like): Estimated probabilities for the positive class.
        model_name (str): Optional name of the model to display in titles.
    
    This function computes the following metrics:
        - Accuracy
        - Precision
        - Recall
        - F1 Score
        - AUC-ROC
      
    It also generates:
        - A confusion matrix (using seaborn heatmap)
        - An ROC curve plot
    """
    # Compute metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    
    # Print out metric values
    print(f"--- {model_name} Evaluation Metrics ---")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"AUC-ROC:   {auc:.4f}")
    
    # Plot confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{model_name} Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    
    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.subplot(1, 2, 2)
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc:.4f})")
    plt.plot([0, 1], [0, 1], 'k--', label="Random Chance")
    plt.title(f"{model_name} ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    
    plt.tight_layout()
    plt.show()
