import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_auc_score
import time

# Create a directory to save plots
plot_dir = 'plots'
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

# Load data
data = load_digits()
X, y = data.data, data.target

# Binarize the output labels for ROC curve calculation
y_binarized = label_binarize(y, classes=np.arange(10))
n_classes = y_binarized.shape[1]

# Models
models = {
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(),
    "k-Nearest Neighbour": KNeighborsClassifier(),
    "SVM": SVC(probability=True)
}

# 5-Fold Cross Validation
skf = StratifiedKFold(n_splits=5)

# Save results
results = {}
roc_data = {}
training_times = {}

for model_name, model in models.items():
    print(f"Training {model_name}...")
    start_time = time.time()
    y_pred = cross_val_predict(model, X, y, cv=skf)
    training_times[model_name] = time.time() - start_time
    conf_matrix = confusion_matrix(y, y_pred)
    accuracy = accuracy_score(y, y_pred)
    results[model_name] = (conf_matrix, accuracy)
    
    # Calculate ROC
    y_pred_proba = cross_val_predict(model, X, y, cv=skf, method='predict_proba')
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_binarized[:, i], y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    roc_data[model_name] = (fpr, tpr, roc_auc)

    # Plot ROC curves and save them in the directory
    plt.figure(figsize=(8, 6))
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(plot_dir, f'{model_name}_ROC.png'))
    plt.close()

# Save results to a text file
with open('results.txt', 'w') as f:
    for model_name, (conf_matrix, accuracy) in results.items():
        f.write(f"\nModel: {model_name}\n")
        f.write(f"Confusion Matrix:\n{conf_matrix}\n")
        f.write(f"Accuracy: {accuracy:.2f}\n")
        f.write(f"Training Time: {training_times[model_name]:.2f} seconds\n")

# Display model speeds
print("\nModel Training Times:")
for model_name, time_taken in sorted(training_times.items(), key=lambda x: x[1]):
    print(f"{model_name}: {time_taken:.2f} seconds")
